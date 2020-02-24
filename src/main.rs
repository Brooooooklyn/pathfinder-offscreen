// pathfinder/examples/canvas_minimal/src/main.rs
//
// Copyright © 2019 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use euclid::default::Size2D;
use gl;
use image::png::PNGEncoder;
use image::ColorType;
use pathfinder_canvas::{CanvasFontContext, CanvasRenderingContext2D, Path2D};
use pathfinder_color::ColorF;
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::{Vector2F, Vector2I};
use pathfinder_gl::{GLDevice, GLVersion};
use pathfinder_gpu::resources::FilesystemResourceLoader;
use pathfinder_gpu::{Device, RenderTarget, TextureData};
use pathfinder_renderer::concurrent::rayon::RayonExecutor;
use pathfinder_renderer::concurrent::scene_proxy::SceneProxy;
use pathfinder_renderer::gpu::options::{DestFramebuffer, RendererOptions};
use pathfinder_renderer::gpu::renderer::Renderer;
use pathfinder_renderer::options::BuildOptions;
use std::env;
use std::fs::File;
use std::path::PathBuf;

use offscreen_gl_context::{
  ColorAttachmentType, GLContext, GLContextAttributes, GLVersion as OffScreenGlVersion,
  NativeGLContext,
};

fn main() {
  let window_size = Vector2I::new(640, 480);
  GLContext::<NativeGLContext>::new_shared_with_dispatcher(
    Size2D::new(window_size.x(), window_size.y()),
    GLContextAttributes::default(),
    ColorAttachmentType::default(),
    sparkle::gl::GlType::Gl,
    OffScreenGlVersion::MajorMinor(4, 1),
    None,
    None,
  )
  .unwrap();

  // gl::load_with(|name| GLContext::<NativeGLContext>::get_proc_address(name) as *const _);

  let shaders_dir = PathBuf::from(env::current_dir().unwrap());
  // Create a Pathfinder renderer.
  let mut renderer = Renderer::new(
    GLDevice::new(GLVersion::GL3, 0),
    &FilesystemResourceLoader {
      directory: shaders_dir,
    },
    DestFramebuffer::full_window(window_size),
    RendererOptions {
      background_color: Some(ColorF::white()),
    },
  );

  // Make a canvas. We're going to draw a house.
  let mut canvas = CanvasRenderingContext2D::new(
    CanvasFontContext::from_system_source(),
    window_size.to_f32(),
  );

  // Set line width.
  canvas.set_line_width(10.0);

  // Draw walls.
  canvas.stroke_rect(RectF::new(
    Vector2F::new(75.0, 140.0),
    Vector2F::new(150.0, 110.0),
  ));

  // Draw door.
  canvas.fill_rect(RectF::new(
    Vector2F::new(130.0, 190.0),
    Vector2F::new(40.0, 60.0),
  ));

  // Draw roof.
  let mut path = Path2D::new();
  path.move_to(Vector2F::new(50.0, 140.0));
  path.line_to(Vector2F::new(150.0, 60.0));
  path.line_to(Vector2F::new(250.0, 140.0));
  path.close_path();
  canvas.stroke_path(path);

  // Render the canvas to screen.
  let scene = SceneProxy::from_scene(canvas.into_scene(), RayonExecutor);
  scene.build_and_render(&mut renderer, BuildOptions::default());
  let viewport = RectI::new(Vector2I::default(), window_size);
  let texture_data_receiver = renderer
    .device
    .read_pixels(&RenderTarget::Default, viewport);

  let pixels = match renderer.device.recv_texture_data(&texture_data_receiver) {
    TextureData::U8(pixels) => pixels,
    _ => panic!("Unexpected pixel format for default framebuffer!"),
  };
  let mut output = File::create("./test.png").unwrap();
  let encoder = PNGEncoder::new(&mut output);
  encoder
    .encode(
      pixels.as_ref(),
      window_size.x() as u32,
      window_size.y() as u32,
      ColorType::Rgba8,
    )
    .unwrap();
}
