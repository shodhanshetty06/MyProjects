import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { KeyRound, Mail, Unplug } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { disconnectGmailAccount, listGmailAccounts } from "@/features/email-tracking/api";
import { SettingsShell } from "@/features/settings/components/SettingsShell";
import { toast } from "@/stores/toastStore";

const passwordSchema = z
  .object({
    currentPassword: z.string().min(1, "Required"),
    newPassword: z.string().min(8, "At least 8 characters"),
    confirmPassword: z.string().min(1, "Required"),
  })
  .refine((d) => d.newPassword === d.confirmPassword, { message: "Passwords don't match", path: ["confirmPassword"] });
type PasswordForm = z.infer<typeof passwordSchema>;

export function SettingsSecurityPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const [submitted, setSubmitted] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<PasswordForm>({ resolver: zodResolver(passwordSchema) });

  const accountsQuery = useQuery({
    queryKey: ["gmail-accounts", userId],
    queryFn: () => listGmailAccounts(userId),
  });

  const disconnectMutation = useMutation({
    mutationFn: (id: string) => disconnectGmailAccount(userId, id),
    onSuccess: () => {
      toast.success("Disconnected");
      void accountsQuery.refetch();
    },
  });

  async function onSubmit() {
    await new Promise((resolve) => setTimeout(resolve, 300));
    setSubmitted(true);
  }

  return (
    <SettingsShell>
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Change password</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {submitted ? (
              <NotConnectedNotice description="Password changes aren't wired up on this backend yet - this form validates end to end, but nothing was actually changed." />
            ) : (
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-3" noValidate>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-ink-secondary">Current password</label>
                  <Input type="password" invalid={!!errors.currentPassword} {...register("currentPassword")} />
                  {errors.currentPassword && <p className="text-xs text-danger">{errors.currentPassword.message}</p>}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-ink-secondary">New password</label>
                    <Input type="password" invalid={!!errors.newPassword} {...register("newPassword")} />
                    {errors.newPassword && <p className="text-xs text-danger">{errors.newPassword.message}</p>}
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-ink-secondary">Confirm new password</label>
                    <Input type="password" invalid={!!errors.confirmPassword} {...register("confirmPassword")} />
                    {errors.confirmPassword && <p className="text-xs text-danger">{errors.confirmPassword.message}</p>}
                  </div>
                </div>
                <Button type="submit" isLoading={isSubmitting}>
                  <KeyRound className="size-4" aria-hidden />
                  Update password
                </Button>
              </form>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Connected accounts</CardTitle>
          </CardHeader>
          <CardContent>
            {accountsQuery.isLoading ? (
              <SkeletonText lines={2} />
            ) : (accountsQuery.data ?? []).length === 0 ? (
              <p className="text-xs text-ink-muted">No connected accounts. Connect Gmail from the application tracker's Tracked Emails tab.</p>
            ) : (
              <ul className="space-y-2">
                {(accountsQuery.data ?? []).map((account) => (
                  <li key={account.id} className="flex items-center gap-3 rounded-[var(--radius-md)] border border-border p-2.5">
                    <Mail className="size-4 text-ink-muted" aria-hidden />
                    <span className="flex-1 truncate text-sm text-ink">{account.email_address}</span>
                    <Button variant="ghost" size="sm" onClick={() => disconnectMutation.mutate(account.id)}>
                      <Unplug className="size-3.5" aria-hidden />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </SettingsShell>
  );
}
