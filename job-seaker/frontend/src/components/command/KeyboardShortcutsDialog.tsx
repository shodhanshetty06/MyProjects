import { Dialog } from "@/components/ui/Dialog";
import { Kbd } from "@/components/ui/Kbd";

const SHORTCUTS: { keys: string[]; description: string }[] = [
  { keys: ["⌘", "K"], description: "Open the command bar" },
  { keys: ["Esc"], description: "Close a dialog or the command bar" },
  { keys: ["↑", "↓"], description: "Move through command bar results" },
  { keys: ["↵"], description: "Select the highlighted result" },
  { keys: ["?"], description: "Show this shortcuts reference" },
];

interface KeyboardShortcutsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function KeyboardShortcutsDialog({ open, onOpenChange }: KeyboardShortcutsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Keyboard shortcuts">
      <ul className="space-y-2.5">
        {SHORTCUTS.map((shortcut) => (
          <li key={shortcut.description} className="flex items-center justify-between gap-4">
            <span className="text-sm text-ink-secondary">{shortcut.description}</span>
            <span className="flex shrink-0 gap-1">
              {shortcut.keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}
