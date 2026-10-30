"""Scene building: determinism, layout constraints and the state queries."""

from __future__ import annotations

import numpy as np
import pytest

from lang2act.scene import (
    CUBE_HALF,
    SceneConfig,
    build_scene,
    is_inside_region,
    is_on_table,
    object_pos,
    sample_layout,
)


def test_layout_is_deterministic_in_the_seed():
    a = sample_layout(SceneConfig(seed=7))
    b = sample_layout(SceneConfig(seed=7))
    assert a.keys() == b.keys()
    for name in a:
        assert np.allclose(a[name], b[name])


def test_different_seeds_give_different_layouts():
    a = sample_layout(SceneConfig(seed=1))
    b = sample_layout(SceneConfig(seed=2))
    assert not np.allclose(a["red_cube"], b["red_cube"])


@pytest.mark.parametrize("seed", range(6))
def test_cubes_are_separated_and_clear_of_the_bowl(seed: int):
    cfg = SceneConfig(seed=seed)
    layout = sample_layout(cfg)
    cubes = [v for k, v in layout.items() if k.endswith("_cube")]
    for i, p in enumerate(cubes):
        assert np.linalg.norm(p - layout["bowl"]) >= cfg.min_cube_bowl_clearance
        for q in cubes[i + 1 :]:
            assert np.linalg.norm(p - q) >= cfg.min_cube_separation


def test_scene_has_the_required_objects():
    _, _, info = build_scene(SceneConfig(seed=0))
    assert set(info.cubes) == {"red_cube", "green_cube", "blue_cube"}
    assert info.regions == ["bowl"]
    assert info.color_of("red_cube") == "red"
    assert info.by_color("green") == ["green_cube"]


def test_cubes_settle_on_the_table():
    model, data, info = build_scene(SceneConfig(seed=3))
    for name in info.cubes:
        z = object_pos(model, data, info.objects[name].body)[2]
        assert z == pytest.approx(CUBE_HALF, abs=2e-3)
        assert is_on_table(model, data, info, name)
        assert not is_inside_region(model, data, info, name, "bowl")


def test_robot_only_model_has_no_free_objects():
    model, _, info = build_scene(SceneConfig(seed=0), with_objects=False)
    assert info.objects == {}
    assert model.nq == 9  # 7 arm joints + 2 fingers, no free joints


def test_region_containment_needs_both_radius_and_rim():
    model, data, info = build_scene(SceneConfig(seed=0))
    bowl = info.objects["bowl"]
    bowl_xy = object_pos(model, data, bowl.body)[:2]
    cube = info.objects["red_cube"]
    joint = model.joint(f"{cube.name}_free")

    # Directly above the bowl but above the rim: not inside.
    data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3] = [*bowl_xy, bowl.rim_z + 0.2]
    import mujoco

    mujoco.mj_forward(model, data)
    assert not is_inside_region(model, data, info, "red_cube", "bowl")

    # Inside the walls and below the rim: inside.
    data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3] = [*bowl_xy, 0.03]
    mujoco.mj_forward(model, data)
    assert is_inside_region(model, data, info, "red_cube", "bowl")
