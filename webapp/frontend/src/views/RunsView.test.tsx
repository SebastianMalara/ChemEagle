import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { useAsyncResource, usePollingResource } from "../api/hooks";
import { mockExperiments, mockMonitor, mockRuns, mockSourceDetail } from "../api/mockData";
import { RunsView } from "./RunsView";

vi.mock("../api/client", () => ({
  api: {
    getRunSource: vi.fn(),
    exportRun: vi.fn(),
  },
}));

vi.mock("../api/hooks", () => ({
  useAsyncResource: vi.fn(),
  usePollingResource: vi.fn(),
}));

describe("RunsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useAsyncResource).mockImplementation((loader, deps) => {
      if (deps.length === 0) {
        return {
          data: mockExperiments,
          error: "",
          loading: false,
          reload: vi.fn(),
        };
      }

      return {
        data: mockRuns,
        error: "",
        loading: false,
        reload: vi.fn(),
      };
    });

    vi.mocked(usePollingResource).mockReturnValue({
      data: mockMonitor,
      error: "",
      loading: false,
      reload: vi.fn(),
    });

    vi.mocked(api.getRunSource).mockResolvedValue(mockSourceDetail);
    vi.mocked(api.exportRun).mockResolvedValue({
      reactions: "./data/exports/run-demo-1_reactions.parquet",
      reaction_molecules: "./data/exports/run-demo-1_reaction_molecules.parquet",
    });
  });

  it("keeps the run tail hidden until explicitly opened", async () => {
    render(<RunsView />);

    const runStatLabel = screen
      .getAllByText("Runs")
      .find((node) => node.classList.contains("stat-label"));

    expect(screen.getByText("Source queue").closest("section")).toHaveClass("batch-pane");
    expect(within(screen.getByText("Reactions").closest(".stat-card")!).getByText("48")).toBeInTheDocument();
    expect(within(runStatLabel?.closest(".stat-card")!).getByText("1")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Open log" })).toBeInTheDocument();
    expect(screen.queryByText(/\[INFO] Extracting scheme_014\.pdf/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open log" }));

    expect(await screen.findByText(/\[INFO] Extracting scheme_014\.pdf/)).toBeInTheDocument();
  });

  it("summarizes exports without dumping raw response json", async () => {
    render(<RunsView />);

    const exportButton = await screen.findByRole("button", { name: "Export selected run" });
    await waitFor(() => expect(exportButton).not.toBeDisabled());

    fireEvent.click(exportButton);

    expect(await screen.findByText("Exported 2 parquet files.")).toBeInTheDocument();
    expect(screen.queryByText(/reaction_molecules/)).not.toBeInTheDocument();
  });
});
