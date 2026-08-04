import VeraGridEngine as gce

grid = gce.open_file("/usr/local/google/home/sxzhou/Downloads/solved_case_sc0.m")

# we need to initialize with a power flow solution
pf_options = gce.PowerFlowOptions()
power_flow = gce.PowerFlowDriver(grid=grid, options=pf_options)
power_flow.run()

# declare the CPF options
vc_options = gce.ContinuationPowerFlowOptions(
    step=0.001,
    approximation_order=gce.CpfParametrization.ArcLength,
    adapt_step=True,
    step_min=0.00001,
    step_max=0.2,
    error_tol=1e-3,
    tol=1e-6,
    max_it=20,
    stop_at=gce.CpfStopAt.Full,
    verbose=False,
)

# We compose the target direction
factor = 2.0
base_power = power_flow.results.Sbus / grid.Sbase
vc_inputs = gce.ContinuationPowerFlowInput(
    Sbase=base_power, Vbase=power_flow.results.voltage, Starget=base_power * factor
)

# declare the CPF driver and run
vc = gce.ContinuationPowerFlowDriver(
    grid=grid, options=vc_options, inputs=vc_inputs, pf_options=pf_options
)
vc.run()
