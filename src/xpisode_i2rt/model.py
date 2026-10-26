"""A YAM's robot model, as xpisode declares models.

i2rt makes the model of an arm with its gripper itself (``combine_arm_and_gripper_xml``: the
arm's MJCF, the gripper merged in at its mount), as a file in /tmp that refers to the meshes
in i2rt's package by absolute paths. A declared model is a self-contained directory, so here
the meshes are copied in, at their paths under i2rt's ``robot_models``, and referred to
relatively. The joints are i2rt's: ``joint1`` to ``joint6`` turn the arm, as the device's
``joint_pos`` reports them; the fingers' slides are not driven by a field, as the gripper's
opening is normalized, not in metres.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

import i2rt.robot_models
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from xpisode_format.models import DeclaredModel, JointMap, declare

ENTRY = "model.xml"
MODELS = Path(i2rt.robot_models.__file__).parent
"""Where i2rt keeps its models; the meshes keep their paths under it."""


def yam_model(
    arm: ArmType, gripper: GripperType, dof: int, stream: str, directory: Path
) -> DeclaredModel:
    """Write the model of ``arm`` with ``gripper`` into ``directory`` and declare it; ``stream``
    is the device's stream with the ``joint_pos`` field of its ``dof`` arm joints."""
    combined = Path(combine_arm_and_gripper_xml(arm, gripper))
    try:
        tree = ET.parse(combined)
    finally:
        combined.unlink(missing_ok=True)  # i2rt leaves one in /tmp for every call
    for element in tree.getroot().iter():
        file = element.get("file")
        if file is None:
            continue
        source = Path(file)
        if not source.is_relative_to(MODELS):
            raise ValueError(f"the model refers to {source}, outside i2rt's robot_models")
        relative = Path("assets") / source.relative_to(MODELS)
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, directory / relative)
        element.set("file", relative.as_posix())
    tree.write(directory / ENTRY, encoding="unicode")
    names = tuple(f"joint{i + 1}" for i in range(dof))
    return declare(
        directory,
        format="mjcf",
        entry=ENTRY,
        joints=(JointMap(stream=stream, field="joint_pos", names=names),),
        source=(
            f"i2rt {metadata.version('i2rt')}: {arm.value} with {gripper.value}, "
            "by combine_arm_and_gripper_xml"
        ),
    )
