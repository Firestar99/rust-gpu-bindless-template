use rust_gpu_bindless_shader_builder::ShaderSymbolsBuilder;

fn main() -> anyhow::Result<()> {
    ShaderSymbolsBuilder::new("my-app-shader", "spirv-unknown-vulkan1.2").build()?;
    Ok(())
}
