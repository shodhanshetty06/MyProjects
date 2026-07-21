import { Plus } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { SettingsShell } from "@/features/settings/components/SettingsShell";

export function SettingsApiKeysPage() {
  return (
    <SettingsShell>
      <Card>
        <CardHeader>
          <CardTitle>API keys</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-ink-secondary">Generate keys to access your data programmatically.</p>
          <NotConnectedNotice description="Self-service API keys aren't wired up on this backend yet - this screen is ready for when that endpoint ships." />
          <Button disabled>
            <Plus className="size-4" aria-hidden />
            Generate new key
          </Button>
        </CardContent>
      </Card>
    </SettingsShell>
  );
}
