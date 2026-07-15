import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * Top-level error boundary. Phase 0: logs to console only — reporting to
 * an observability backend will be wired in a later phase.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info);
  }

  render() {
    if (this.state.hasError) {
      return <div role="alert">Something went wrong.</div>;
    }
    return this.props.children;
  }
}
