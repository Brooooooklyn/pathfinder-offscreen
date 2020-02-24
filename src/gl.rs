// pathfinder/gl/src/lib.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An OpenGL implementation of the device abstraction.

use half::f16;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::Vector2I;
use pathfinder_gpu::resources::ResourceLoader;
use pathfinder_gpu::{BlendFactor, BlendOp, BufferData, BufferTarget, BufferUploadMode, ClearOps};
use pathfinder_gpu::{DepthFunc, Device, Primitive, RenderOptions, RenderState, RenderTarget};
use pathfinder_gpu::{ShaderKind, StencilFunc, TextureData, TextureDataRef, TextureFormat};
use pathfinder_gpu::{UniformData, VertexAttrClass, VertexAttrDescriptor, VertexAttrType};
use pathfinder_simd::default::F32x4;
use sparkle::gl;
use sparkle::gl::types::{GLboolean, GLchar, GLenum, GLfloat, GLint, GLsizei, GLsizeiptr, GLsync};
use sparkle::gl::types::{GLuint, GLvoid};
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::rc::Rc;
use std::str;
use std::time::Duration;

pub struct GLDevice {
  version: GLVersion,
  default_framebuffer: GLuint,
  gl_: Rc<sparkle::gl::Gl>,
}

impl GLDevice {
  #[inline]
  pub fn new(
    version: GLVersion,
    default_framebuffer: GLuint,
    gl_: Rc<sparkle::gl::Gl>,
  ) -> GLDevice {
    GLDevice {
      version,
      default_framebuffer,
      gl_,
    }
  }

  pub fn set_default_framebuffer(&mut self, framebuffer: GLuint) {
    self.default_framebuffer = framebuffer;
  }

  fn set_texture_parameters(&self, texture: &GLTexture) {
    self.bind_texture(texture, 0);
    self
      .gl_
      .tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
    ck(self.gl_.as_ref());
    self
      .gl_
      .tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
    ck(self.gl_.as_ref());
    self.gl_.tex_parameter_i(
      gl::TEXTURE_2D,
      gl::TEXTURE_WRAP_S,
      gl::CLAMP_TO_EDGE as GLint,
    );
    ck(self.gl_.as_ref());
    self.gl_.tex_parameter_i(
      gl::TEXTURE_2D,
      gl::TEXTURE_WRAP_T,
      gl::CLAMP_TO_EDGE as GLint,
    );
    ck(self.gl_.as_ref());
  }

  fn set_render_state(&self, render_state: &RenderState<GLDevice>) {
    self.bind_render_target(render_state.target);

    let (origin, size) = (render_state.viewport.origin(), render_state.viewport.size());
    self
      .gl_
      .viewport(origin.x(), origin.y(), size.x(), size.y());

    if render_state.options.clear_ops.has_ops() {
      self.clear(&render_state.options.clear_ops);
    }

    self.use_program(render_state.program);
    self.bind_vertex_array(render_state.vertex_array);
    for (texture_unit, texture) in render_state.textures.iter().enumerate() {
      self.bind_texture(texture, texture_unit as u32);
    }

    render_state
      .uniforms
      .iter()
      .for_each(|(uniform, data)| self.set_uniform(uniform, data));
    self.set_render_options(&render_state.options);
  }

  fn set_render_options(&self, render_options: &RenderOptions) {
    // Set blend.
    match render_options.blend {
      None => {
        self.gl_.disable(gl::BLEND);
        ck(self.gl_.as_ref());
      }
      Some(ref blend) => {
        self.gl_.blend_func_separate(
          blend.src_rgb_factor.to_gl_blend_factor(),
          blend.dest_rgb_factor.to_gl_blend_factor(),
          blend.src_alpha_factor.to_gl_blend_factor(),
          blend.dest_alpha_factor.to_gl_blend_factor(),
        );
        ck(self.gl_.as_ref());
        self.gl_.blend_equation(blend.op.to_gl_blend_op());
        ck(self.gl_.as_ref());
        self.gl_.enable(gl::BLEND);
        ck(self.gl_.as_ref());
      }
    }

    // Set depth.
    match render_options.depth {
      None => {
        self.gl_.disable(gl::DEPTH_TEST);
        ck(self.gl_.as_ref());
      }
      Some(ref state) => {
        self.gl_.depth_func(state.func.to_gl_depth_func());
        ck(self.gl_.as_ref());
        self.gl_.depth_mask(state.write as bool);
        ck(self.gl_.as_ref());
        self.gl_.enable(gl::DEPTH_TEST);
        ck(self.gl_.as_ref());
      }
    }

    // Set stencil.
    match render_options.stencil {
      None => {
        self.gl_.disable(gl::STENCIL_TEST);
        ck(self.gl_.as_ref());
      }
      Some(ref state) => {
        self.gl_.stencil_func(
          state.func.to_gl_stencil_func(),
          state.reference as GLint,
          state.mask,
        );
        ck(self.gl_.as_ref());
        let (pass_action, write_mask) = if state.write {
          (gl::REPLACE, state.mask)
        } else {
          (gl::KEEP, 0)
        };
        self.gl_.stencil_op(gl::KEEP, gl::KEEP, pass_action);
        ck(self.gl_.as_ref());
        self.gl_.stencil_mask(write_mask);
        self.gl_.enable(gl::STENCIL_TEST);
        ck(self.gl_.as_ref());
      }
    }

    // Set color mask.
    let color_mask = render_options.color_mask as bool;
    self
      .gl_
      .color_mask(color_mask, color_mask, color_mask, color_mask);
    ck(self.gl_.as_ref());
  }

  fn set_uniform(&self, uniform: &GLUniform, data: &UniformData) {
    unsafe {
      match *data {
        UniformData::Float(value) => {
          self.gl_.uniform_1f(uniform.location, value);
          ck(self.gl_.as_ref());
        }
        UniformData::Int(value) => {
          self.gl_.uniform_1i(uniform.location, value);
          ck(self.gl_.as_ref());
        }
        UniformData::Mat2(data) => {
          assert_eq!(mem::size_of::<F32x4>(), 4 * 4);
          self.gl_.uniform_matrix_2fv(
            uniform.location,
            false,
            &[data[0], data[1], data[2], data[3]],
          );
        }
        UniformData::Mat4(data) => {
          assert_eq!(mem::size_of::<[F32x4; 4]>(), 4 * 4 * 4);
          let data_ptr: *const F32x4 = data.as_ptr();
          self.gl_.uniform_matrix_4fv(
            uniform.location,
            false,
            std::slice::from_raw_parts(data_ptr as *const f32, 64),
          );
        }
        UniformData::Vec2(data) => {
          self.gl_.uniform_2f(uniform.location, data.x(), data.y());
          ck(self.gl_.as_ref());
        }
        UniformData::Vec4(data) => {
          self
            .gl_
            .uniform_4f(uniform.location, data.x(), data.y(), data.z(), data.w());
          ck(self.gl_.as_ref());
        }
        UniformData::TextureUnit(unit) => {
          self.gl_.uniform_1i(uniform.location, unit as GLint);
          ck(self.gl_.as_ref());
        }
      }
    }
  }

  fn reset_render_state(&self, render_state: &RenderState<GLDevice>) {
    self.reset_render_options(&render_state.options);
    for texture_unit in 0..(render_state.textures.len() as u32) {
      self.unbind_texture(texture_unit);
    }
    self.unuse_program();
    self.unbind_vertex_array();
  }

  fn reset_render_options(&self, render_options: &RenderOptions) {
    if render_options.blend.is_some() {
      self.gl_.disable(gl::BLEND);
      ck(self.gl_.as_ref());
    }

    if render_options.depth.is_some() {
      self.gl_.disable(gl::DEPTH_TEST);
      ck(self.gl_.as_ref());
    }

    if render_options.stencil.is_some() {
      self.gl_.stencil_mask(!0);
      ck(self.gl_.as_ref());
      self.gl_.disable(gl::STENCIL_TEST);
      ck(self.gl_.as_ref());
    }
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE) },
      gl::Gl::Gles(gles) => unsafe { gles.ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE) },
    };
    ck(self.gl_.as_ref());
  }

  #[inline]
  fn gen_texture(&self) -> GLuint {
    let mut texture = 0;
    unsafe {
      match self.gl_.as_ref() {
        gl::Gl::Gl(gl) => gl.GenTextures(1, &mut texture),
        gl::Gl::Gles(gles) => gles.GenTextures(1, &mut texture),
      };
    };
    texture
  }
}

impl Device for GLDevice {
  type Buffer = GLBuffer;
  type Framebuffer = GLFramebuffer;
  type Program = GLProgram;
  type Shader = GLShader;
  type Texture = GLTexture;
  type TextureDataReceiver = GLTextureDataReceiver;
  type TimerQuery = GLTimerQuery;
  type Uniform = GLUniform;
  type VertexArray = GLVertexArray;
  type VertexAttr = GLVertexAttr;

  fn create_texture(&self, format: TextureFormat, size: Vector2I) -> GLTexture {
    let texture = self.gen_texture();
    let texture = GLTexture {
      gl_texture: texture,
      size,
      format,
    };
    ck(self.gl_.as_ref());
    self.bind_texture(&texture, 0);
    self.gl_.tex_image_2d(
      gl::TEXTURE_2D,
      0,
      format.gl_internal_format(),
      size.x() as GLsizei,
      size.y() as GLsizei,
      0,
      format.gl_format(),
      format.gl_type(),
      None,
    );
    ck(self.gl_.as_ref());

    self.set_texture_parameters(&texture);
    texture
  }

  fn create_texture_from_data(
    &self,
    format: TextureFormat,
    size: Vector2I,
    data: TextureDataRef,
  ) -> GLTexture {
    let len = match data {
      TextureDataRef::F16(d) => d.len() * 2,
      TextureDataRef::F32(d) => d.len() * 4,
      TextureDataRef::U8(d) => d.len(),
    };
    let data_ptr = data.check_and_extract_data_ptr(size, format);
    unsafe {
      let texture = self.gen_texture();
      let texture = GLTexture {
        gl_texture: texture,
        size,
        format: TextureFormat::R8,
      };
      ck(self.gl_.as_ref());
      self.bind_texture(&texture, 0);
      self.gl_.tex_image_2d(
        gl::TEXTURE_2D,
        0,
        format.gl_internal_format(),
        size.x() as GLsizei,
        size.y() as GLsizei,
        0,
        format.gl_format(),
        format.gl_type(),
        Some(std::slice::from_raw_parts(data_ptr as *const _, len)),
      );
      self.set_texture_parameters(&texture);
      texture
    }
  }

  fn create_shader_from_source(&self, name: &str, source: &[u8], kind: ShaderKind) -> GLShader {
    // FIXME(pcwalton): Do this once and cache it.
    let glsl_version_spec = self.version.to_glsl_version_spec();

    let mut output = vec![];
    self.preprocess(&mut output, source, glsl_version_spec);
    let source = output;

    let gl_shader_kind = match kind {
      ShaderKind::Vertex => gl::VERTEX_SHADER,
      ShaderKind::Fragment => gl::FRAGMENT_SHADER,
    };

    unsafe {
      let gl_shader = self.gl_.create_shader(gl_shader_kind);
      ck(self.gl_.as_ref());
      self.gl_.shader_source(gl_shader, &[source.as_slice()]);
      ck(self.gl_.as_ref());
      self.gl_.compile_shader(gl_shader);
      ck(self.gl_.as_ref());

      let mut compile_status = vec![0];
      self
        .gl_
        .get_shader_iv(gl_shader, gl::COMPILE_STATUS, &mut compile_status);
      ck(self.gl_.as_ref());
      if compile_status[0] != gl::TRUE as GLint {
        let mut info_log_length = vec![0];
        self
          .gl_
          .get_shader_iv(gl_shader, gl::INFO_LOG_LENGTH, &mut info_log_length);
        ck(self.gl_.as_ref());
        let info_log = self.gl_.get_shader_info_log(gl_shader);
        ck(self.gl_.as_ref());
        error!("Shader info log:\n{}", info_log);
        panic!("{:?} shader '{}' compilation failed", kind, name);
      }

      GLShader {
        gl_: self.gl_.clone(),
        gl_shader,
      }
    }
  }

  fn create_program_from_shaders(
    &self,
    _resources: &dyn ResourceLoader,
    name: &str,
    vertex_shader: GLShader,
    fragment_shader: GLShader,
  ) -> GLProgram {
    let gl_program;
    unsafe {
      gl_program = self.gl_.create_program();
      ck(self.gl_.as_ref());
      self.gl_.attach_shader(gl_program, vertex_shader.gl_shader);
      ck(self.gl_.as_ref());
      self
        .gl_
        .attach_shader(gl_program, fragment_shader.gl_shader);
      ck(self.gl_.as_ref());
      self.gl_.link_program(gl_program);
      ck(self.gl_.as_ref());

      let mut link_status = vec![0];
      self
        .gl_
        .get_program_iv(gl_program, gl::LINK_STATUS, &mut link_status);
      ck(self.gl_.as_ref());
      if link_status[0] != gl::TRUE as GLint {
        let info_log_length = 0;
        self
          .gl_
          .get_program_iv(gl_program, gl::INFO_LOG_LENGTH, &mut [info_log_length]);
        ck(self.gl_.as_ref());
        let info_log = self.gl_.get_program_info_log(gl_program);
        ck(self.gl_.as_ref());
        eprintln!("Program info log:\n{}", info_log);
        panic!("Program '{}' linking failed", name);
      }
    }

    GLProgram {
      gl_: self.gl_.clone(),
      gl_program,
      vertex_shader,
      fragment_shader,
    }
  }

  #[inline]
  fn create_vertex_array(&self) -> GLVertexArray {
    let mut gl_vertex_array = 0;
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.GenVertexArrays(1, &mut gl_vertex_array) },
      gl::Gl::Gles(gles) => unsafe { gles.GenVertexArrays(1, &mut gl_vertex_array) },
    };
    ck(self.gl_.as_ref());
    GLVertexArray {
      gl_: self.gl_.clone(),
      gl_vertex_array,
    }
  }

  fn get_vertex_attr(&self, program: &Self::Program, name: &str) -> Option<GLVertexAttr> {
    let attr = self
      .gl_
      .get_attrib_location(program.gl_program, &format!("a{}", name));
    ck(self.gl_.as_ref());
    if attr < 0 {
      None
    } else {
      Some(GLVertexAttr {
        gl_: self.gl_.clone(),
        attr: attr as GLuint,
      })
    }
  }

  fn get_uniform(&self, program: &GLProgram, name: &str) -> GLUniform {
    let location = self.gl_.get_uniform_location(program.gl_program, name);
    ck(self.gl_.as_ref());
    GLUniform { location }
  }

  fn configure_vertex_attr(
    &self,
    vertex_array: &GLVertexArray,
    attr: &GLVertexAttr,
    descriptor: &VertexAttrDescriptor,
  ) {
    debug_assert_ne!(descriptor.stride, 0);

    self.bind_vertex_array(vertex_array);

    unsafe {
      let attr_type = descriptor.attr_type.to_gl_type();
      match descriptor.class {
        VertexAttrClass::Float | VertexAttrClass::FloatNorm => {
          let normalized = descriptor.class == VertexAttrClass::FloatNorm;
          self.gl_.vertex_attrib_pointer(
            attr.attr,
            descriptor.size as GLint,
            attr_type,
            normalized,
            descriptor.stride as GLint,
            descriptor.offset as u32,
          );
          ck(self.gl_.as_ref());
        }
        VertexAttrClass::Int => {
          match self.gl_.as_ref() {
            gl::Gl::Gl(gl) => gl.VertexAttribIPointer(
              attr.attr,
              descriptor.size as GLint,
              attr_type,
              descriptor.stride as GLint,
              descriptor.offset as *const GLvoid,
            ),
            gl::Gl::Gles(gles) => gles.VertexAttribIPointer(
              attr.attr,
              descriptor.size as GLint,
              attr_type,
              descriptor.stride as GLint,
              descriptor.offset as *const GLvoid,
            ),
          };
          ck(self.gl_.as_ref());
        }
      }

      self
        .gl_
        .vertex_attrib_divisor(attr.attr, descriptor.divisor);
      ck(self.gl_.as_ref());
      self.gl_.enable_vertex_attrib_array(attr.attr);
      ck(self.gl_.as_ref());
    }

    self.unbind_vertex_array();
  }

  fn create_framebuffer(&self, texture: GLTexture) -> GLFramebuffer {
    let mut gl_framebuffer = 0;
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.GenFramebuffers(1, &mut gl_framebuffer) },
      gl::Gl::Gles(gles) => unsafe { gles.GenFramebuffers(1, &mut gl_framebuffer) },
    };
    ck(self.gl_.as_ref());
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.BindFramebuffer(gl::FRAMEBUFFER, gl_framebuffer) },
      gl::Gl::Gles(gles) => unsafe { gles.BindFramebuffer(gl::FRAMEBUFFER, gl_framebuffer) },
    };
    ck(self.gl_.as_ref());
    self.bind_texture(&texture, 0);
    self.gl_.framebuffer_texture_2d(
      gl::FRAMEBUFFER,
      gl::COLOR_ATTACHMENT0,
      gl::TEXTURE_2D,
      texture.gl_texture,
      0,
    );
    ck(self.gl_.as_ref());
    assert_eq!(
      self.gl_.check_framebuffer_status(gl::FRAMEBUFFER),
      gl::FRAMEBUFFER_COMPLETE
    );
    GLFramebuffer {
      gl_: self.gl_.clone(),
      gl_framebuffer,
      texture,
    }
  }

  fn create_buffer(&self) -> GLBuffer {
    let mut gl_buffer = 0;
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.GenBuffers(1, &mut gl_buffer) },
      gl::Gl::Gles(gles) => unsafe { gles.GenBuffers(1, &mut gl_buffer) },
    };
    ck(self.gl_.as_ref());
    GLBuffer {
      gl_: self.gl_.clone(),
      gl_buffer,
    }
  }

  fn allocate_buffer<T>(
    &self,
    buffer: &GLBuffer,
    data: BufferData<T>,
    target: BufferTarget,
    mode: BufferUploadMode,
  ) {
    let target = match target {
      BufferTarget::Vertex => gl::ARRAY_BUFFER,
      BufferTarget::Index => gl::ELEMENT_ARRAY_BUFFER,
    };
    let (ptr, len) = match data {
      BufferData::Uninitialized(len) => (ptr::null(), len),
      BufferData::Memory(buffer) => (buffer.as_ptr() as *const GLvoid, buffer.len()),
    };
    let len = (len * mem::size_of::<T>()) as GLsizeiptr;
    let usage = mode.to_gl_usage();
    unsafe {
      self.gl_.bind_buffer(target, buffer.gl_buffer);
      ck(self.gl_.as_ref());
      self.gl_.buffer_data(target, len, ptr, usage);
      ck(self.gl_.as_ref());
    }
  }

  #[inline]
  fn framebuffer_texture<'f>(&self, framebuffer: &'f Self::Framebuffer) -> &'f Self::Texture {
    &framebuffer.texture
  }

  #[inline]
  fn destroy_framebuffer(&self, framebuffer: Self::Framebuffer) -> Self::Texture {
    let texture = GLTexture {
      gl_texture: framebuffer.texture.gl_texture,
      size: framebuffer.texture.size,
      format: framebuffer.texture.format,
    };
    mem::forget(framebuffer);
    texture
  }

  #[inline]
  fn texture_format(&self, texture: &Self::Texture) -> TextureFormat {
    texture.format
  }

  #[inline]
  fn texture_size(&self, texture: &Self::Texture) -> Vector2I {
    texture.size
  }

  fn upload_to_texture(&self, texture: &Self::Texture, rect: RectI, data: TextureDataRef) {
    let len = match data {
      TextureDataRef::F16(d) => d.len() * 2,
      TextureDataRef::F32(d) => d.len() * 4,
      TextureDataRef::U8(d) => d.len(),
    };
    let data_ptr = data.check_and_extract_data_ptr(rect.size(), texture.format);
    assert!(rect.size().x() >= 0);
    assert!(rect.size().y() >= 0);
    assert!(rect.max_x() <= texture.size.x());
    assert!(rect.max_y() <= texture.size.y());
    unsafe {
      let data = std::slice::from_raw_parts(data_ptr as *const _, len);
      self.bind_texture(texture, 0);
      if rect.origin() == Vector2I::default() && rect.size() == texture.size {
        self.gl_.tex_image_2d(
          gl::TEXTURE_2D,
          0,
          texture.format.gl_internal_format(),
          texture.size.x() as GLsizei,
          texture.size.y() as GLsizei,
          0,
          texture.format.gl_format(),
          texture.format.gl_type(),
          Some(data),
        );
        ck(self.gl_.as_ref());
      } else {
        self.gl_.tex_sub_image_2d(
          gl::TEXTURE_2D,
          0,
          rect.origin().x(),
          rect.origin().y(),
          rect.size().x() as GLsizei,
          rect.size().y() as GLsizei,
          texture.format.gl_format(),
          texture.format.gl_type(),
          data,
        );
        ck(self.gl_.as_ref());
      }
    }

    self.set_texture_parameters(texture);
  }

  fn read_pixels(
    &self,
    render_target: &RenderTarget<GLDevice>,
    viewport: RectI,
  ) -> GLTextureDataReceiver {
    let (origin, size) = (viewport.origin(), viewport.size());
    let format = self.render_target_format(render_target);
    self.bind_render_target(render_target);
    let byte_size = size.x() as usize * size.y() as usize * format.bytes_per_pixel() as usize;

    let mut gl_pixel_buffer = 0;
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.GenBuffers(1, &mut gl_pixel_buffer) },
      gl::Gl::Gles(gles) => unsafe { gles.GenBuffers(1, &mut gl_pixel_buffer) },
    };
    ck(self.gl_.as_ref());
    self.gl_.bind_buffer(gl::PIXEL_PACK_BUFFER, gl_pixel_buffer);
    ck(self.gl_.as_ref());
    unsafe {
      self.gl_.buffer_data(
        gl::PIXEL_PACK_BUFFER,
        byte_size as GLsizeiptr,
        ptr::null(),
        gl::STATIC_READ,
      )
    };
    ck(self.gl_.as_ref());

    self.gl_.read_pixels(
      origin.x(),
      origin.y(),
      size.x() as GLsizei,
      size.y() as GLsizei,
      format.gl_format(),
      format.gl_type(),
    );
    ck(self.gl_.as_ref());

    let gl_sync = self.gl_.fence_sync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);

    GLTextureDataReceiver {
      gl_: self.gl_.clone(),
      gl_pixel_buffer,
      gl_sync,
      size,
      format,
    }
  }

  fn begin_commands(&self) {
    // TODO(pcwalton): Add some checks in debug mode to make sure render commands are bracketed
    // by these?
  }

  fn end_commands(&self) {
    self.gl_.flush();
  }

  fn draw_arrays(&self, index_count: u32, render_state: &RenderState<Self>) {
    self.set_render_state(render_state);
    self.gl_.draw_arrays(
      render_state.primitive.to_gl_primitive(),
      0,
      index_count as GLsizei,
    );
    ck(self.gl_.as_ref());
    self.reset_render_state(render_state);
  }

  fn draw_elements(&self, index_count: u32, render_state: &RenderState<Self>) {
    self.set_render_state(render_state);
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe {
        gl.DrawElements(
          render_state.primitive.to_gl_primitive(),
          index_count as GLsizei,
          gl::UNSIGNED_INT,
          ptr::null(),
        )
      },
      gl::Gl::Gles(gles) => unsafe {
        gles.DrawElements(
          render_state.primitive.to_gl_primitive(),
          index_count as GLsizei,
          gl::UNSIGNED_INT,
          ptr::null(),
        )
      },
    }
    // ck(self.gl_.as_ref());
    self.reset_render_state(render_state);
  }

  fn draw_elements_instanced(
    &self,
    index_count: u32,
    instance_count: u32,
    render_state: &RenderState<Self>,
  ) {
    self.set_render_state(render_state);
    unsafe {
      self.gl_.draw_elements_instanced(
        render_state.primitive.to_gl_primitive(),
        index_count as GLsizei,
        gl::UNSIGNED_INT,
        0,
        instance_count as GLsizei,
      );
      ck(self.gl_.as_ref());
    }
    self.reset_render_state(render_state);
  }

  #[inline]
  fn create_timer_query(&self) -> GLTimerQuery {
    let mut gl_query = 0;
    match self.gl_.as_ref() {
      gl::Gl::Gl(gl) => unsafe { gl.GenQueries(1, &mut gl_query) },
      gl::Gl::Gles(gles) => unsafe { gles.GenQueriesEXT(1, &mut gl_query) },
    };
    ck(self.gl_.as_ref());
    GLTimerQuery {
      gl_: self.gl_.clone(),
      gl_query,
    }
  }

  #[inline]
  fn begin_timer_query(&self, query: &Self::TimerQuery) {
    self.gl_.begin_query(gl::TIME_ELAPSED, query.gl_query);
    ck(self.gl_.as_ref());
  }

  #[inline]
  fn end_timer_query(&self, _: &Self::TimerQuery) {
    unsafe {
      self.gl_.end_query(gl::TIME_ELAPSED);
      ck(self.gl_.as_ref());
    }
  }

  fn try_recv_timer_query(&self, query: &Self::TimerQuery) -> Option<Duration> {
    unsafe {
      let result = self
        .gl_
        .get_query_object_iv(query.gl_query, gl::QUERY_RESULT_AVAILABLE);
      ck(self.gl_.as_ref());
      if result == gl::FALSE as GLint {
        None
      } else {
        Some(self.recv_timer_query(query))
      }
    }
  }

  fn recv_timer_query(&self, query: &Self::TimerQuery) -> Duration {
    unsafe {
      let result = self
        .gl_
        .get_query_object_ui64v(query.gl_query, gl::QUERY_RESULT);
      ck(self.gl_.as_ref());
      Duration::from_nanos(result)
    }
  }

  fn try_recv_texture_data(&self, receiver: &Self::TextureDataReceiver) -> Option<TextureData> {
    unsafe {
      let result = self
        .gl_
        .client_wait_sync(receiver.gl_sync, gl::SYNC_FLUSH_COMMANDS_BIT, 0);
      ck(self.gl_.as_ref());
      if result == gl::TIMEOUT_EXPIRED || result == gl::WAIT_FAILED {
        None
      } else {
        Some(self.get_texture_data(receiver))
      }
    }
  }

  fn recv_texture_data(&self, receiver: &Self::TextureDataReceiver) -> TextureData {
    unsafe {
      let result = self
        .gl_
        .client_wait_sync(receiver.gl_sync, gl::SYNC_FLUSH_COMMANDS_BIT, !0);
      ck(self.gl_.as_ref());
      debug_assert!(result != gl::TIMEOUT_EXPIRED && result != gl::WAIT_FAILED);
      self.get_texture_data(receiver)
    }
  }

  #[inline]
  fn bind_buffer(&self, vertex_array: &GLVertexArray, buffer: &GLBuffer, target: BufferTarget) {
    self.bind_vertex_array(vertex_array);
    unsafe {
      self
        .gl_
        .bind_buffer(target.to_gl_target(), buffer.gl_buffer);
      ck(self.gl_.as_ref());
    }
    self.unbind_vertex_array();
  }

  #[inline]
  fn create_shader(
    &self,
    resources: &dyn ResourceLoader,
    name: &str,
    kind: ShaderKind,
  ) -> Self::Shader {
    let suffix = match kind {
      ShaderKind::Vertex => 'v',
      ShaderKind::Fragment => 'f',
    };
    let path = format!("shaders/gl3/{}.{}s.glsl", name, suffix);
    println!("{:?}", &path);
    self.create_shader_from_source(name, &resources.slurp(&path).unwrap(), kind)
  }
}

impl GLDevice {
  fn bind_render_target(&self, attachment: &RenderTarget<GLDevice>) {
    match *attachment {
      RenderTarget::Default => self.bind_default_framebuffer(),
      RenderTarget::Framebuffer(framebuffer) => self.bind_framebuffer(framebuffer),
    }
  }

  fn bind_vertex_array(&self, vertex_array: &GLVertexArray) {
    unsafe {
      self.gl_.bind_vertex_array(vertex_array.gl_vertex_array);
      ck(self.gl_.as_ref());
    }
  }

  fn unbind_vertex_array(&self) {
    unsafe {
      self.gl_.bind_vertex_array(0);
      ck(self.gl_.as_ref());
    }
  }

  fn bind_texture(&self, texture: &GLTexture, unit: u32) {
    unsafe {
      self.gl_.active_texture(gl::TEXTURE0 + unit);
      ck(self.gl_.as_ref());
      self.gl_.bind_texture(gl::TEXTURE_2D, texture.gl_texture);
      ck(self.gl_.as_ref());
    }
  }

  fn unbind_texture(&self, unit: u32) {
    unsafe {
      self.gl_.active_texture(gl::TEXTURE0 + unit);
      ck(self.gl_.as_ref());
      self.gl_.bind_texture(gl::TEXTURE_2D, 0);
      ck(self.gl_.as_ref());
    }
  }

  fn use_program(&self, program: &GLProgram) {
    unsafe {
      self.gl_.use_program(program.gl_program);
      ck(self.gl_.as_ref());
    }
  }

  fn unuse_program(&self) {
    unsafe {
      self.gl_.use_program(0);
      ck(self.gl_.as_ref());
    }
  }

  fn bind_default_framebuffer(&self) {
    unsafe {
      self
        .gl_
        .bind_framebuffer(gl::FRAMEBUFFER, self.default_framebuffer);
      ck(self.gl_.as_ref());
    }
  }

  fn bind_framebuffer(&self, framebuffer: &GLFramebuffer) {
    unsafe {
      self
        .gl_
        .bind_framebuffer(gl::FRAMEBUFFER, framebuffer.gl_framebuffer);
      ck(self.gl_.as_ref());
    }
  }

  fn preprocess(&self, output: &mut Vec<u8>, source: &[u8], version: &str) {
    let mut index = 0;
    while index < source.len() {
      if source[index..].starts_with(b"{{") {
        let end_index = source[index..]
          .iter()
          .position(|character| *character == b'}')
          .expect("Expected `}`!")
          + index;
        assert_eq!(source[end_index + 1], b'}');
        let ident = String::from_utf8_lossy(&source[(index + 2)..end_index]);
        if ident == "version" {
          output.extend_from_slice(version.as_bytes());
        } else {
          panic!("unknown template variable: `{}`", ident);
        }
        index = end_index + 2;
      } else {
        output.push(source[index]);
        index += 1;
      }
    }
  }

  fn clear(&self, ops: &ClearOps) {
    let mut flags = 0;
    if let Some(color) = ops.color {
      self.gl_.color_mask(true, true, true, true);
      ck(self.gl_.as_ref());
      self
        .gl_
        .clear_color(color.r(), color.g(), color.b(), color.a());
      ck(self.gl_.as_ref());
      flags |= gl::COLOR_BUFFER_BIT;
    }
    if let Some(depth) = ops.depth {
      self.gl_.depth_mask(true);
      ck(self.gl_.as_ref());
      self.gl_.clear_depth(depth as _);
      ck(self.gl_.as_ref()); // FIXME(pcwalton): GLES
      flags |= gl::DEPTH_BUFFER_BIT;
    }
    if let Some(stencil) = ops.stencil {
      self.gl_.stencil_mask(!0);
      ck(self.gl_.as_ref());
      self.gl_.clear_stencil(stencil as GLint);
      ck(self.gl_.as_ref());
      flags |= gl::STENCIL_BUFFER_BIT;
    }
    // if flags != 0 {
    //   self.gl_.clear(flags);
    //   ck(self.gl_.as_ref());
    // }
  }

  fn render_target_format(&self, render_target: &RenderTarget<GLDevice>) -> TextureFormat {
    match *render_target {
      RenderTarget::Default => TextureFormat::RGBA8,
      RenderTarget::Framebuffer(ref framebuffer) => self.framebuffer_texture(framebuffer).format,
    }
  }

  fn get_texture_data(&self, receiver: &GLTextureDataReceiver) -> TextureData {
    unsafe {
      let (format, size) = (receiver.format, receiver.size);
      let channels = format.channels();
      let (mut texture_data, texture_data_ptr, texture_data_len);
      match format {
        TextureFormat::R8 | TextureFormat::RGBA8 => {
          let mut pixels: Vec<u8> = vec![0; size.x() as usize * size.y() as usize * channels];
          texture_data_ptr = pixels.as_mut_ptr();
          texture_data_len = pixels.len() * mem::size_of::<u8>();
          texture_data = TextureData::U8(pixels);
        }
        TextureFormat::R16F | TextureFormat::RGBA16F => {
          let mut pixels: Vec<f16> =
            vec![f16::default(); size.x() as usize * size.y() as usize * channels];
          texture_data_ptr = pixels.as_mut_ptr() as *mut u8;
          texture_data_len = pixels.len() * mem::size_of::<f16>();
          texture_data = TextureData::F16(pixels);
        }
        TextureFormat::RGBA32F => {
          let mut pixels = vec![0.0; size.x() as usize * size.y() as usize * channels];
          texture_data_ptr = pixels.as_mut_ptr() as *mut u8;
          texture_data_len = pixels.len() * mem::size_of::<f32>();
          texture_data = TextureData::F32(pixels);
        }
      }

      self
        .gl_
        .bind_buffer(gl::PIXEL_PACK_BUFFER, receiver.gl_pixel_buffer);
      ck(self.gl_.as_ref());
      match self.gl_.as_ref() {
        gl::Gl::Gl(gl) => {
          gl.GetBufferSubData(
            gl::PIXEL_PACK_BUFFER,
            0,
            texture_data_len as GLsizeiptr,
            texture_data_ptr as *mut GLvoid,
          );
        }
        gl::Gl::Gles(gles) => {
          gles.BufferSubData(
            gl::PIXEL_PACK_BUFFER,
            0,
            texture_data_len as GLsizeiptr,
            texture_data_ptr as *mut GLvoid,
          );
        }
      };
      ck(self.gl_.as_ref());
      self.gl_.bind_buffer(gl::PIXEL_PACK_BUFFER, 0);
      ck(self.gl_.as_ref());

      match texture_data {
        TextureData::U8(ref mut pixels) => flip_y(pixels, size, channels),
        TextureData::U16(ref mut pixels) => flip_y(pixels, size, channels),
        TextureData::F16(ref mut pixels) => flip_y(pixels, size, channels),
        TextureData::F32(ref mut pixels) => flip_y(pixels, size, channels),
      }

      texture_data
    }
  }
}

pub struct GLVertexArray {
  gl_: Rc<gl::Gl>,
  pub gl_vertex_array: GLuint,
}

impl Drop for GLVertexArray {
  #[inline]
  fn drop(&mut self) {
    self.gl_.delete_vertex_arrays(&mut [self.gl_vertex_array]);
    ck(self.gl_.as_ref());
  }
}

pub struct GLVertexAttr {
  gl_: Rc<gl::Gl>,
  attr: GLuint,
}

impl GLVertexAttr {
  pub fn configure_float(
    &self,
    size: GLint,
    gl_type: GLuint,
    normalized: bool,
    stride: GLsizei,
    offset: usize,
    divisor: GLuint,
  ) {
    self
      .gl_
      .vertex_attrib_pointer(self.attr, size, gl_type, normalized, stride, offset as u32);
    ck(self.gl_.as_ref());
    self.gl_.vertex_attrib_divisor(self.attr, divisor);
    ck(self.gl_.as_ref());
    self.gl_.enable_vertex_attrib_array(self.attr);
    ck(self.gl_.as_ref());
  }

  pub fn configure_int(
    &self,
    size: GLint,
    gl_type: GLuint,
    stride: GLsizei,
    offset: usize,
    divisor: GLuint,
  ) {
    unsafe {
      match self.gl_.as_ref() {
        gl::Gl::Gl(gl) => {
          gl.VertexAttribIPointer(self.attr, size, gl_type, stride, offset as *const GLvoid)
        }
        gl::Gl::Gles(gles) => {
          gles.VertexAttribIPointer(self.attr, size, gl_type, stride, offset as *const GLvoid)
        }
      };
      ck(self.gl_.as_ref());
      self.gl_.vertex_attrib_divisor(self.attr, divisor);
      ck(self.gl_.as_ref());
      self.gl_.enable_vertex_attrib_array(self.attr);
      ck(self.gl_.as_ref());
    }
  }
}

pub struct GLFramebuffer {
  gl_: Rc<gl::Gl>,
  pub gl_framebuffer: GLuint,
  pub texture: GLTexture,
}

impl Drop for GLFramebuffer {
  fn drop(&mut self) {
    self.gl_.delete_framebuffers(&mut [self.gl_framebuffer]);
    ck(self.gl_.as_ref());
  }
}

pub struct GLBuffer {
  gl_: Rc<gl::Gl>,
  pub gl_buffer: GLuint,
}

impl Drop for GLBuffer {
  fn drop(&mut self) {
    unsafe {
      self.gl_.delete_buffers(&mut [self.gl_buffer]);
      ck(self.gl_.as_ref());
    }
  }
}

#[derive(Debug)]
pub struct GLUniform {
  location: GLint,
}

pub struct GLProgram {
  gl_: Rc<gl::Gl>,
  pub gl_program: GLuint,
  #[allow(dead_code)]
  vertex_shader: GLShader,
  #[allow(dead_code)]
  fragment_shader: GLShader,
}

impl Drop for GLProgram {
  fn drop(&mut self) {
    self.gl_.delete_program(self.gl_program);
    ck(self.gl_.as_ref());
  }
}

pub struct GLShader {
  gl_: Rc<gl::Gl>,
  gl_shader: GLuint,
}

impl Drop for GLShader {
  fn drop(&mut self) {
    self.gl_.delete_shader(self.gl_shader);
    ck(self.gl_.as_ref());
  }
}

pub struct GLTexture {
  gl_texture: GLuint,
  pub size: Vector2I,
  pub format: TextureFormat,
}

pub struct GLTimerQuery {
  gl_: Rc<gl::Gl>,
  gl_query: GLuint,
}

impl Drop for GLTimerQuery {
  #[inline]
  fn drop(&mut self) {
    self.gl_.delete_queries(&mut [self.gl_query]);
    ck(self.gl_.as_ref());
  }
}

trait BlendFactorExt {
  fn to_gl_blend_factor(self) -> GLenum;
}

impl BlendFactorExt for BlendFactor {
  #[inline]
  fn to_gl_blend_factor(self) -> GLenum {
    match self {
      BlendFactor::Zero => gl::ZERO,
      BlendFactor::One => gl::ONE,
      BlendFactor::SrcAlpha => gl::SRC_ALPHA,
      BlendFactor::OneMinusSrcAlpha => gl::ONE_MINUS_SRC_ALPHA,
      BlendFactor::DestAlpha => gl::DST_ALPHA,
      BlendFactor::OneMinusDestAlpha => gl::ONE_MINUS_DST_ALPHA,
      BlendFactor::DestColor => gl::DST_COLOR,
    }
  }
}

trait BlendOpExt {
  fn to_gl_blend_op(self) -> GLenum;
}

impl BlendOpExt for BlendOp {
  #[inline]
  fn to_gl_blend_op(self) -> GLenum {
    match self {
      BlendOp::Add => gl::FUNC_ADD,
      BlendOp::Subtract => gl::FUNC_SUBTRACT,
      BlendOp::ReverseSubtract => gl::FUNC_REVERSE_SUBTRACT,
      BlendOp::Min => gl::MIN,
      BlendOp::Max => gl::MAX,
    }
  }
}

trait BufferTargetExt {
  fn to_gl_target(self) -> GLuint;
}

impl BufferTargetExt for BufferTarget {
  fn to_gl_target(self) -> GLuint {
    match self {
      BufferTarget::Vertex => gl::ARRAY_BUFFER,
      BufferTarget::Index => gl::ELEMENT_ARRAY_BUFFER,
    }
  }
}

trait BufferUploadModeExt {
  fn to_gl_usage(self) -> GLuint;
}

impl BufferUploadModeExt for BufferUploadMode {
  fn to_gl_usage(self) -> GLuint {
    match self {
      BufferUploadMode::Static => gl::STATIC_DRAW,
      BufferUploadMode::Dynamic => gl::DYNAMIC_DRAW,
    }
  }
}

trait DepthFuncExt {
  fn to_gl_depth_func(self) -> GLenum;
}

impl DepthFuncExt for DepthFunc {
  fn to_gl_depth_func(self) -> GLenum {
    match self {
      DepthFunc::Less => gl::LESS,
      DepthFunc::Always => gl::ALWAYS,
    }
  }
}

trait PrimitiveExt {
  fn to_gl_primitive(self) -> GLuint;
}

impl PrimitiveExt for Primitive {
  fn to_gl_primitive(self) -> GLuint {
    match self {
      Primitive::Triangles => gl::TRIANGLES,
      Primitive::Lines => gl::LINES,
    }
  }
}

trait StencilFuncExt {
  fn to_gl_stencil_func(self) -> GLenum;
}

impl StencilFuncExt for StencilFunc {
  fn to_gl_stencil_func(self) -> GLenum {
    match self {
      StencilFunc::Always => gl::ALWAYS,
      StencilFunc::Equal => gl::EQUAL,
    }
  }
}

trait TextureFormatExt {
  fn gl_internal_format(self) -> GLint;
  fn gl_format(self) -> GLuint;
  fn gl_type(self) -> GLuint;
}

impl TextureFormatExt for TextureFormat {
  fn gl_internal_format(self) -> GLint {
    match self {
      TextureFormat::R8 => gl::R8 as GLint,
      TextureFormat::R16F => gl::R16F as GLint,
      TextureFormat::RGBA8 => gl::RGBA as GLint,
      TextureFormat::RGBA16F => gl::RGBA16F as GLint,
      TextureFormat::RGBA32F => gl::RGBA32F as GLint,
    }
  }

  fn gl_format(self) -> GLuint {
    match self {
      TextureFormat::R8 | TextureFormat::R16F => gl::RED,
      TextureFormat::RGBA8 | TextureFormat::RGBA16F | TextureFormat::RGBA32F => gl::RGBA,
    }
  }

  fn gl_type(self) -> GLuint {
    match self {
      TextureFormat::R8 | TextureFormat::RGBA8 => gl::UNSIGNED_BYTE,
      TextureFormat::R16F | TextureFormat::RGBA16F => gl::HALF_FLOAT,
      TextureFormat::RGBA32F => gl::FLOAT,
    }
  }
}

trait VertexAttrTypeExt {
  fn to_gl_type(self) -> GLuint;
}

impl VertexAttrTypeExt for VertexAttrType {
  fn to_gl_type(self) -> GLuint {
    match self {
      VertexAttrType::F32 => gl::FLOAT,
      VertexAttrType::I16 => gl::SHORT,
      VertexAttrType::I8 => gl::BYTE,
      VertexAttrType::U16 => gl::UNSIGNED_SHORT,
      VertexAttrType::U8 => gl::UNSIGNED_BYTE,
    }
  }
}

pub struct GLTextureDataReceiver {
  gl_: Rc<gl::Gl>,
  gl_pixel_buffer: GLuint,
  gl_sync: GLsync,
  size: Vector2I,
  format: TextureFormat,
}

impl Drop for GLTextureDataReceiver {
  fn drop(&mut self) {
    self.gl_.delete_buffers(&mut [self.gl_pixel_buffer]);
    ck(self.gl_.as_ref());
    self.gl_.delete_sync(self.gl_sync);
    ck(self.gl_.as_ref());
  }
}

/// The version/dialect of OpenGL we should render with.
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum GLVersion {
  /// OpenGL 3.0+, core profile.
  GL3 = 0,
  /// OpenGL ES 3.0+.
  GLES3 = 1,
}

impl GLVersion {
  fn to_glsl_version_spec(&self) -> &'static str {
    match *self {
      GLVersion::GL3 => "330",
      GLVersion::GLES3 => "300 es",
    }
  }
}

// Error checking

#[cfg(debug_assertions)]
fn ck(gl_: &gl::Gl) {
  // Note that ideally we should be calling gl::GetError() in a loop until it
  // returns gl::NO_ERROR, but for now we'll just report the first one we find.
  let err = gl_.get_error();
  if err != gl::NO_ERROR {
    panic!(
      "GL error: 0x{:x} ({})",
      err,
      match err {
        gl::INVALID_ENUM => "INVALID_ENUM",
        gl::INVALID_VALUE => "INVALID_VALUE",
        gl::INVALID_OPERATION => "INVALID_OPERATION",
        gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
        gl::OUT_OF_MEMORY => "OUT_OF_MEMORY",
        gl::STACK_UNDERFLOW => "STACK_UNDERFLOW",
        gl::STACK_OVERFLOW => "STACK_OVERFLOW",
        _ => "Unknown",
      }
    );
  }
}

#[cfg(not(debug_assertions))]
fn ck(gl_: &gl::Gl) {}

// Utilities

// Flips a buffer of image data upside-down.
fn flip_y<T>(pixels: &mut [T], size: Vector2I, channels: usize) {
  let stride = size.x() as usize * channels;
  for y in 0..(size.y() as usize / 2) {
    let (index_a, index_b) = (y * stride, (size.y() as usize - y - 1) * stride);
    for offset in 0..stride {
      pixels.swap(index_a + offset, index_b + offset);
    }
  }
}
