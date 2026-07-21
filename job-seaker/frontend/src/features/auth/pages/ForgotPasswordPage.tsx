import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowLeft, Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { forgotPasswordSchema, type ForgotPasswordFormValues } from "@/features/auth/schemas";
import { paths } from "@/routes/paths";

export function ForgotPasswordPage() {
  const [submitted, setSubmitted] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ForgotPasswordFormValues>({ resolver: zodResolver(forgotPasswordSchema) });

  async function onSubmit() {
    await new Promise((resolve) => setTimeout(resolve, 350));
    setSubmitted(true);
  }

  return (
    <div>
      <h1 className="text-lg font-semibold text-ink">Reset your password</h1>
      <p className="mt-1 text-sm text-ink-secondary">
        Enter the email on your account and we'll send a reset link.
      </p>

      {submitted ? (
        <NotConnectedNotice
          className="mt-6"
          description="Password reset isn't wired up on this backend yet - this screen is built and validates end to end, but no email will actually arrive."
        />
      ) : (
        <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4" noValidate>
          <div className="space-y-1.5">
            <label htmlFor="email" className="text-xs font-medium text-ink-secondary">
              Email
            </label>
            <Input id="email" type="email" autoComplete="email" invalid={!!errors.email} {...register("email")} />
            {errors.email && <p className="text-xs text-danger">{errors.email.message}</p>}
          </div>

          <Button type="submit" className="w-full" isLoading={isSubmitting}>
            Send reset link
            <Send className="size-4" aria-hidden />
          </Button>
        </form>
      )}

      <Link
        to={paths.login}
        className="mt-6 flex items-center justify-center gap-1.5 text-sm font-medium text-ink-secondary hover:text-ink"
      >
        <ArrowLeft className="size-3.5" aria-hidden />
        Back to log in
      </Link>
    </div>
  );
}
