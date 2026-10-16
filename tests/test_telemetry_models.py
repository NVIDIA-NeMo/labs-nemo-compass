# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep telemetry field meanings discoverable through the canonical JSON schemas."""

import inspect

import pytest
from pydantic import ValidationError

from insight_agent import telemetry_models


@pytest.mark.parametrize(
    "model",
    [
        value
        for _, value in inspect.getmembers(telemetry_models, inspect.isclass)
        if issubclass(value, telemetry_models.TelemetryModel)
        and value is not telemetry_models.TelemetryModel
    ],
)
def test_phase_contract_fields_have_descriptions_including_nested_models(model):
    schema = model.model_json_schema()
    definitions = [schema, *schema.get("$defs", {}).values()]
    for definition in definitions:
        for name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{definition['title']}.{name} has no meaning defined"


def test_phase_contract_rejects_unknown_status_and_fields():
    with pytest.raises(ValidationError):
        telemetry_models.SignalOutput.model_validate(
            {"status": "success", "problems": (), "limitations": ()}
        )
    with pytest.raises(ValidationError):
        telemetry_models.SourceCodeValidationOutput.model_validate({"unexpected": 1})


def test_run_input_fields_have_descriptions_including_nested_config():
    from insight_agent.telemetry import RunInput

    schema = RunInput.model_json_schema()
    for definition in [schema, *schema.get("$defs", {}).values()]:
        for name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{definition['title']}.{name}"


@pytest.mark.parametrize("model", telemetry_models.SIGNAL_INPUT_TYPES.values())
def test_signal_configuration_fields_are_annotated(model):
    test_phase_contract_fields_have_descriptions_including_nested_models(model)
