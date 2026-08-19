import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { UploadPanel } from "./UploadPanel";


describe("UploadPanel", () => {
  it("submits the selected hybrid mode", async () => {
    const user = userEvent.setup();
    const createExtraction = vi.fn().mockResolvedValue({
      id: "job-1",
      mode: "hybrid",
      status: "queued",
      warnings: [],
    });

    render(
      <UploadPanel
        createExtraction={createExtraction}
        onSubmitted={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText("Hybrid"));
    await user.upload(
      screen.getByLabelText("Timing diagram"),
      new File(["image"], "wave.png", { type: "image/png" }),
    );
    await user.click(screen.getByRole("button", { name: "Extract" }));

    await waitFor(() => {
      expect(createExtraction).toHaveBeenCalledWith(expect.any(File), "hybrid");
    });
  });
});
