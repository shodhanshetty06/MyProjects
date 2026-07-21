import { zodResolver } from "@hookform/resolvers/zod";
import { KeyRound } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { resetPasswordSchema, type ResetPasswordFormValues } from "@/features/auth/schemas";
import { paths } from "@/routes/paths";

export function ResetPasswordPage() {
  const [submitted, setSubmitted] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ResetPasswordFormValues>({ resolver: zodResolver(resetPasswordSchema) });

  async function onSubmit() {
    await new Promise((resolve) => setTimeout(resolve, 350));
    setSubmitted(true);
  }

  return (
    <div>
      <h1 className="text-lg font-semibold text-ink">Set a new password</h1>
      <p className="mt-1 text-sm text-ink-secondary">Choose a new password for your account.</p>

      {submitted ? (
        <NotConnectedNotice
          className="mt-6"
          description="Password reset isn't wired up on this backend yet - this screen validates and is ready to go, but nothing was actually changed."
        />
      ) : (
        <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4" noValidate>
          <div className="space-y-1.5">
            <label htmlFor="password" className="text-xs font-medium text-ink-secondary">
              New password
            </label>
            <Input
              id="password"
              type="password"
              autoComplete="new-password"
              invalid={!!errors.password}
              {...register("password")}
            />
            {errors.password && <p className="text-xs text-danger">{errors.password.message}</p>}
          </div>

          <div className="space-y-1.5">
            <label htmlFor="confirmPassword" className="text-xs font-medium text-ink-secondary">
              Confirm new password
            </label>
            <Input
              id="confirmPassword"
              type="password"
              autoComplete="new-password"
              invalid={!!errors.confirmPassword}
              {...register("confirmPassword")}
            />
            {errors.confirmPassword && <p className="text-xs text-danger">{errors.confirmPassword.message}</p>}
          </div>

          <Button type="submit" className="w-full" isLoading={isSubmitting}>
            Reset password
            <KeyRound className="size-4" aria-hidden />
          </Button>
        </form>
      )}

      <Link to={paths.login} className="mt-6 block text-center text-sm font-medium text-ink-secondary hover:text-ink">
        Back to log in
      </Link>
    </div>
  );
}
