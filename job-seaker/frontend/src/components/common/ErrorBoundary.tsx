import { AlertTriangle, RotateCcw } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";

import { Button } from "@/components/ui/Button";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * Top-level error boundary. Logs to console only - reporting to an
 * observability backend is a backend-side concern out of scope here.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false });
    window.location.href = "/";
  };

  render() {
    if (this.state.hasError) {
      return (
        <div role="alert" className="flex min-h-screen flex-col items-center justify-center gap-4 bg-bg px-6 text-center">
          <div className="flex size-12 items-center justify-center rounded-full bg-danger-soft text-danger">
            <AlertTriangle className="size-6" aria-hidden />
          </div>
          <div className="space-y-1">
            <p className="text-lg font-semibold text-ink">Something went wrong</p>
            <p className="max-w-sm text-sm text-ink-secondary">
              An unexpected error interrupted this page. You can try reloading, or head back home.
            </p>
          </div>
          <Button onClick={this.handleReset}>
            <RotateCcw className="size-4" aria-hidden />
            Back to safety
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}
