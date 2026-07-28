import { Command } from "cmdk";
import {
  BarChart3,
  Bell,
  Briefcase,
  Building2,
  Calendar,
  FileText,
  History,
  Keyboard,
  LayoutDashboard,
  Mail,
  MessageSquareText,
  Moon,
  Pin,
  Search,
  Send,
  Settings,
  Shield,
  Sparkles,
  Sun,
  Target,
  Wand2,
} from "lucide-react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { paths } from "@/routes/paths";
import { useThemeStore } from "@/stores/themeStore";
import { useUiStore } from "@/stores/uiStore";

interface CommandEntry {
  id: string;
  label: string;
  group: string;
  icon: typeof Search;
  shortcut?: string;
  run: (navigate: ReturnType<typeof useNavigate>) => void;
}

const ENTRIES: CommandEntry[] = [
  { id: "dashboard", label: "Go to Dashboard", group: "Navigate", icon: LayoutDashboard, run: (nav) => nav(paths.dashboard) },
  { id: "search", label: "Search jobs", group: "Navigate", icon: Search, run: (nav) => nav(paths.search) },
  { id: "saved-jobs", label: "Saved jobs", group: "Navigate", icon: Pin, run: (nav) => nav(paths.savedJobs) },
  { id: "recommended-jobs", label: "Recommended jobs", group: "Navigate", icon: Sparkles, run: (nav) => nav(paths.search) },
  { id: "companies", label: "Explore companies", group: "Navigate", icon: Building2, run: (nav) => nav(paths.companies) },
  { id: "resumes", label: "Resume manager", group: "Navigate", icon: FileText, run: (nav) => nav(paths.resumes) },
  { id: "ats-generator", label: "ATS resume generator", group: "Navigate", icon: Wand2, run: (nav) => nav(paths.atsGenerator) },
  { id: "auto-apply", label: "Auto apply tasks", group: "Navigate", icon: Send, run: (nav) => nav(paths.autoApply) },
  { id: "tracker", label: "Application tracker", group: "Navigate", icon: Briefcase, run: (nav) => nav(paths.tracker) },
  { id: "tracker-calendar", label: "Tracker calendar", group: "Navigate", icon: Calendar, run: (nav) => nav(paths.trackerCalendar) },
  { id: "tracker-emails", label: "Tracked emails", group: "Navigate", icon: Mail, run: (nav) => nav(paths.trackerEmails) },
  { id: "coach", label: "AI career coach", group: "Navigate", icon: MessageSquareText, run: (nav) => nav(paths.coach) },
  { id: "coach-chat", label: "Chat with AI coach", group: "Navigate", icon: MessageSquareText, run: (nav) => nav(paths.coachChat) },
  { id: "coach-insights", label: "Coach insight history", group: "Navigate", icon: History, run: (nav) => nav(paths.coachInsights) },
  { id: "analytics", label: "Analytics", group: "Navigate", icon: BarChart3, run: (nav) => nav(paths.analytics) },
  { id: "notifications", label: "Notifications", group: "Navigate", icon: Bell, run: (nav) => nav(paths.notifications) },
  { id: "settings", label: "Settings", group: "Navigate", icon: Settings, run: (nav) => nav(paths.settings) },
  { id: "admin", label: "Admin portal", group: "Navigate", icon: Shield, run: (nav) => nav(paths.admin) },
];

export function useCommandPaletteHotkey() {
  const setOpen = useUiStore((state) => state.setCommandPaletteOpen);
  const open = useUiStore((state) => state.commandPaletteOpen);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const isMeta = event.metaKey || event.ctrlKey;
      if (isMeta && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(!open);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, setOpen]);
}

export function CommandPalette() {
  const open = useUiStore((state) => state.commandPaletteOpen);
  const setOpen = useUiStore((state) => state.setCommandPaletteOpen);
  const setShortcutsOpen = useUiStore((state) => state.setShortcutsOpen);
  const preference = useThemeStore((state) => state.preference);
  const setPreference = useThemeStore((state) => state.setPreference);
  const navigate = useNavigate();

  const groups = Array.from(new Set(ENTRIES.map((e) => e.group)));

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command palette"
      className="flex flex-col bg-transparent"
      contentClassName="fixed left-1/2 top-[18vh] z-[110] w-full max-w-lg -translate-x-1/2 overflow-hidden rounded-[var(--radius-xl)] border border-border bg-surface shadow-lg outline-none"
      overlayClassName="fixed inset-0 z-[105] bg-overlay backdrop-blur-[2px]"
    >
      <div className="flex items-center gap-2 border-b border-border px-4">
        <Search className="size-4 text-ink-muted" aria-hidden />
        <Command.Input
          placeholder="Search actions, pages, jobs..."
          className="h-12 w-full bg-transparent text-sm text-ink outline-none placeholder:text-ink-muted"
        />
      </div>
      <Command.List className="max-h-80 overflow-y-auto p-2">
        <Command.Empty className="px-3 py-6 text-center text-sm text-ink-muted">No results found.</Command.Empty>
        {groups.map((group) => (
          <Command.Group
            key={group}
            heading={group}
            className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-ink-muted"
          >
            {ENTRIES.filter((e) => e.group === group).map((entry) => (
              <Command.Item
                key={entry.id}
                onSelect={() => {
                  entry.run(navigate);
                  setOpen(false);
                }}
                className="flex cursor-pointer items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm text-ink data-[selected=true]:bg-surface-2"
              >
                <entry.icon className="size-4 text-ink-muted" aria-hidden />
                {entry.label}
              </Command.Item>
            ))}
          </Command.Group>
        ))}
        <Command.Group
          heading="Preferences"
          className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-ink-muted"
        >
          <Command.Item
            onSelect={() => {
              setPreference(preference === "dark" ? "light" : "dark");
              setOpen(false);
            }}
            className="flex cursor-pointer items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm text-ink data-[selected=true]:bg-surface-2"
          >
            {preference === "dark" ? (
              <Sun className="size-4 text-ink-muted" aria-hidden />
            ) : (
              <Moon className="size-4 text-ink-muted" aria-hidden />
            )}
            Toggle theme
          </Command.Item>
          <Command.Item
            onSelect={() => {
              navigate(paths.settingsProfile);
              setOpen(false);
            }}
            className="flex cursor-pointer items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm text-ink data-[selected=true]:bg-surface-2"
          >
            <Target className="size-4 text-ink-muted" aria-hidden />
            Edit profile
          </Command.Item>
          <Command.Item
            onSelect={() => {
              setShortcutsOpen(true);
              setOpen(false);
            }}
            className="flex cursor-pointer items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm text-ink data-[selected=true]:bg-surface-2"
          >
            <Keyboard className="size-4 text-ink-muted" aria-hidden />
            Keyboard shortcuts
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
