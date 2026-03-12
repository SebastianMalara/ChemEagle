import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mockConfig, mockSubmission } from "../api/mockData";
import { api } from "../api/client";
import { IngestView } from "./IngestView";

vi.mock("../api/client", () => ({
  api: {
    getConfig: vi.fn(),
    uploadFiles: vi.fn(),
    submitLiveExperiment: vi.fn(),
    submitSideloadExperiment: vi.fn(),
  },
}));

describe("IngestView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getConfig).mockResolvedValue(mockConfig);
    vi.mocked(api.uploadFiles).mockResolvedValue([]);
    vi.mocked(api.submitLiveExperiment).mockResolvedValue(mockSubmission);
    vi.mocked(api.submitSideloadExperiment).mockResolvedValue(mockSubmission);
  });

  it("shows a compact handoff instead of raw response json", async () => {
    render(
      <MemoryRouter>
        <IngestView />
      </MemoryRouter>,
    );

    await screen.findByDisplayValue("Paperbreaker Review Experiment");

    expect(screen.getByText("Queue live experiment").closest("form")).toHaveClass("batch-pane");

    fireEvent.click(screen.getByRole("button", { name: "Queue live run" }));

    expect(await screen.findByRole("link", { name: "Open runs monitor" })).toBeInTheDocument();
    expect(screen.getByText("Live batch queued")).toBeInTheDocument();
    expect(screen.getByText("Monitor progress and logs from Runs.")).toBeInTheDocument();
    expect(screen.queryByText("Last response")).not.toBeInTheDocument();
    expect(screen.queryByText(/"status_text"/)).not.toBeInTheDocument();
  });
});
