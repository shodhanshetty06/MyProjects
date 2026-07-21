import { Toaster } from "@/components/ui/Toaster";
import { TooltipProvider } from "@/components/ui/Tooltip";
import { AppRoutes } from "@/routes/AppRoutes";

export function App() {
  return (
    <TooltipProvider>
      <AppRoutes />
      <Toaster />
    </TooltipProvider>
  );
}
