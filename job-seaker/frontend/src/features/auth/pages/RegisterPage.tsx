import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { PasswordInput } from "@/components/ui/PasswordInput";
import { useAuth } from "@/features/auth/AuthProvider";
import { registerSchema, type RegisterFormValues } from "@/features/auth/schemas";
import { resolvePostLoginPath } from "@/features/onboarding/api";
import { errorMessage } from "@/lib/errorMessage";
import { paths } from "@/routes/paths";

export function RegisterPage() {
  const { register: registerUser } = useAuth();
  const navigate = useNavigate();
  const [formError, setFormError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<RegisterFormValues>({ resolver: zodResolver(registerSchema) });

  async function onSubmit(values: RegisterFormValues) {
    setFormError(null);
    try {
      await registerUser({ email: values.email, password: values.password });
      // Brand-new accounts have no `UserProfile` row yet, so this always
      // resolves to `/onboarding` — going through the same status check as
      // login keeps there being exactly one place that decides this.
      navigate(await resolvePostLoginPath(), { replace: true });
    } catch (error) {
      setFormError(errorMessage(error, "Couldn't create your account. Try a different email."));
    }
  }

  return (
    <div>
      <h1 className="text-lg font-semibold text-ink">Create your account</h1>
      <p className="mt-1 text-sm text-ink-secondary">Free to start - upload a resume and get matched in minutes.</p>

      <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4" noValidate>
        <div className="space-y-1.5">
          <label htmlFor="email" className="text-xs font-medium text-ink-secondary">
            Email
          </label>
          <Input id="email" type="email" autoComplete="email" invalid={!!errors.email} {...register("email")} />
          {errors.email && <p className="text-xs text-danger">{errors.email.message}</p>}
        </div>

        <div className="space-y-1.5">
          <label htmlFor="password" className="text-xs font-medium text-ink-secondary">
            Password
          </label>
          <PasswordInput
            id="password"
            autoComplete="new-password"
            invalid={!!errors.password}
            {...register("password")}
          />
          {errors.password && <p className="text-xs text-danger">{errors.password.message}</p>}
        </div>

        <div className="space-y-1.5">
          <label htmlFor="confirmPassword" className="text-xs font-medium text-ink-secondary">
            Confirm password
          </label>
          <PasswordInput
            id="confirmPassword"
            autoComplete="new-password"
            invalid={!!errors.confirmPassword}
            {...register("confirmPassword")}
          />
          {errors.confirmPassword && <p className="text-xs text-danger">{errors.confirmPassword.message}</p>}
        </div>

        {formError && (
          <p role="alert" className="text-sm text-danger">
            {formError}
          </p>
        )}

        <Button type="submit" className="w-full" isLoading={isSubmitting}>
          Create account
          <ArrowRight className="size-4" aria-hidden />
        </Button>
      </form>

      <p className="mt-6 text-center text-sm text-ink-secondary">
        Already have an account?{" "}
        <Link to={paths.login} className="font-medium text-brand hover:underline">
          Log in
        </Link>
      </p>
    </div>
  );
}
