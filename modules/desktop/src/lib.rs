// Copyright (C) 2019-2020  Pierre Krieger
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

mod rasterizer;

pub struct Desktop {
    imgui: imgui::Context,
    rasterizer: rasterizer::Rasterizer,
}

impl Desktop {
    pub fn new(dimensions: [u32; 2]) -> Self {
        let mut rasterizer = rasterizer::Rasterizer::new(dimensions);

        let mut imgui = imgui::Context::create();
        // TODO: clipboard
        imgui.io_mut().display_framebuffer_scale = [1.0, 1.0];
        imgui.io_mut().display_size = [dimensions[0] as f32, dimensions[1] as f32];
        imgui.io_mut().font_global_scale = 1.0;
        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    size_pixels: 14.0,
                    ..Default::default()
                }),
            }]);

        imgui.set_renderer_name(Some(imgui::ImString::from(format!(
            "imgui-software-renderer"
        ))));
        let texture_id = rasterizer.add_texture(&imgui.fonts().build_alpha8_texture());
        imgui.fonts().tex_id = texture_id;

        Desktop { imgui, rasterizer }
    }

    /// Returns a buffer containing the RGB pixels.
    pub fn pixels(&self) -> &[u8] {
        self.rasterizer.pixels()
    }

    pub fn render(&mut self) {
        let ui = self.imgui.frame();

        imgui::Window::new(imgui::im_str!("Hello world"))
            .size([300.0, 110.0], imgui::Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(imgui::im_str!("Hello world!"));
                ui.text(imgui::im_str!("こんにちは世界！"));
                ui.text(imgui::im_str!("This...is...imgui-rs!"));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });

        let draw_data = ui.render();
        self.rasterizer.draw(&draw_data);
    }
}
