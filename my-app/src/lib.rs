use rust_gpu_bindless::platform::ash::Debuggers;

pub mod shader;
pub mod simple_compute;

/// the global setting on which debugger to use for integration tests
pub fn debugger() -> Debuggers {
	Debuggers::Validation
}
