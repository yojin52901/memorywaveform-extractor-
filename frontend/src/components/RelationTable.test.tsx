import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { TimingParameter } from "../api/extractions";
import { RelationTable } from "./RelationTable";


describe("RelationTable", () => {
  it("reports the selected timing parameter", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const parameter: TimingParameter = {
      id: "tp_twc",
      name: "tWC",
      from_event_id: "evt_start",
      to_event_id: "evt_end",
      participant_signal_ids: ["sig_we"],
      meaning: "Write-cycle interval.",
      confidence: 0.9,
    };

    render(
      <RelationTable
        parameters={[parameter]}
        relations={[]}
        onSelect={onSelect}
      />,
    );

    await user.click(screen.getByRole("button", { name: "tWC" }));

    expect(onSelect).toHaveBeenCalledWith("tp_twc");
  });

  it("marks low-confidence parameters as review-only", () => {
    const parameter: TimingParameter = {
      id: "tp_twc",
      name: "tWC",
      from_event_id: "evt_start",
      to_event_id: "evt_end",
      participant_signal_ids: ["sig_we"],
      meaning: "Write-cycle interval.",
      confidence: 0.2,
    };

    render(
      <RelationTable
        parameters={[parameter]}
        relations={[]}
        warnings={[
          {
            code: "LOW_CONFIDENCE_RELATION",
            message: "Review tWC.",
            related_ids: ["tp_twc"],
          },
        ]}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("Review required")).toBeTruthy();
  });
});
