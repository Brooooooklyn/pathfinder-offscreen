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
use image::png::PNGEncoder;
use image::ColorType;
use pathfinder_canvas::{CanvasFontContext, CanvasRenderingContext2D, Path2D};
use pathfinder_color::ColorF;
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::rect::RectI;
use pathfinder_geometry::vector::{Vector2F, Vector2I};
use pathfinder_gl::{GLDevice, GLVersion as pathfinder_glversion};
use pathfinder_gpu::{Device, RenderTarget, TextureData};
use pathfinder_renderer::concurrent::rayon::RayonExecutor;
use pathfinder_renderer::concurrent::scene_proxy::SceneProxy;
use pathfinder_renderer::gpu::options::{DestFramebuffer, RendererOptions};
use pathfinder_renderer::gpu::renderer::Renderer;
use pathfinder_renderer::options::BuildOptions;
use pathfinder_resources::embedded::EmbeddedResourceLoader;
use std::fs::File;

use surfman::{
  Connection, ContextAttributeFlags, ContextAttributes, GLVersion, SurfaceAccess, SurfaceType,
};

fn main() {
  let window_size = Vector2I::new(640, 480);
  let connection = Connection::new().unwrap();

  let adapter = connection.create_hardware_adapter().unwrap();

  let mut device = connection.create_device(&adapter).unwrap();

  let context_attributes = ContextAttributes {
    version: GLVersion::new(3, 3),
    flags: ContextAttributeFlags::empty(),
  };
  let context_descriptor = device
    .create_context_descriptor(&context_attributes)
    .unwrap();
  let mut context = device.create_context(&context_descriptor).unwrap();
  let surface = device
    .create_surface(
      &context,
      SurfaceAccess::GPUOnly,
      SurfaceType::Generic {
        size: Size2D::new(window_size.x(), window_size.y()),
      },
    )
    .unwrap();
  device
    .bind_surface_to_context(&mut context, surface)
    .unwrap();

  gl::load_with(|symbol_name| device.get_proc_address(&context, symbol_name));
  device.make_context_current(&context).unwrap();
  // Create a Pathfinder renderer.
  let mut renderer = Renderer::new(
    GLDevice::new(pathfinder_glversion::GL3, 0),
    &EmbeddedResourceLoader::new(),
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
