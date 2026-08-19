def step_optimizer(optimizer, optimize=True):
    """Step an optimizer if it exists, then clear its gradients."""
    if optimizer is not None:
        if optimize:
            optimizer.step()
        optimizer.zero_grad()


def zero_all_grads(estimator):
    """Zero every optimizer's gradients so none survive into the next iteration."""
    gaussians = estimator.scene.gaussians
    for optimizer in [
        estimator.state_optimizer,
        estimator.material_optimizer,
        gaussians.optimizer,
        gaussians.x_optimizer,
        gaussians.scale_optimizer,
    ]:
        if optimizer is not None:
            optimizer.zero_grad()


def step_all_optimizers(estimator, actual_frames):
    """Step every optimizer, each behind its own readiness gate."""
    gaussians = estimator.scene.gaussians

    # Velocity and gravity are unobservable from a single rolled-out frame.
    step_optimizer(estimator.state_optimizer, optimize=actual_frames > 1)

    # Material only becomes observable once the object has been in contact.
    if estimator.check_collision_with_delay(actual_frames):
        step_optimizer(estimator.material_optimizer)

    step_optimizer(gaussians.optimizer)
    step_optimizer(gaussians.x_optimizer)

    # Scene scale is unconstrained until the trajectory bends or hits the ground.
    step_optimizer(
        gaussians.scale_optimizer,
        optimize=actual_frames >= 3 or estimator.check_collision(),
    )
