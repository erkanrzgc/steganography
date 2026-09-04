import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { App } from "./main";

describe("local authentication screen", () => {
  beforeEach(() => sessionStorage.clear());

  it("explains the local token boundary", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Open your workspace" })).toBeTruthy();
    expect(screen.getByLabelText("Local API token")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Authenticate/ })).toBeTruthy();
  });
});
