import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("renders the review-led navigation shell", () => {
    render(
      <MemoryRouter initialEntries={["/review"]}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByText("ChemEagle Web")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Review" })).toBeInTheDocument();
    expect(screen.getByText("Reaction review without setup friction.")).toBeInTheDocument();
  });
});
