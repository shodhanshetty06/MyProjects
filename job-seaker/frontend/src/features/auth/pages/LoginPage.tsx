import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { PasswordInput } from "@/components/ui/PasswordInput";
import { useAuth } from "@/features/auth/AuthProvider";
import { loginSchema, type LoginFormValues } from "@/features/auth/schemas";
import { resolvePostLoginPath } from "@/features/onboarding/api";
import { errorMessage } from "@/lib/errorMessage";
import { paths } from "@/routes/paths";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [formError, setFormError] = useState<string | null>(null);

  const redirectTo = (location.state as { from?: { pathname?: string } } | null)?.from?.pathname;

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormValues>({ resolver: zodResolver(loginSchema) });

  async function onSubmit(values: LoginFormValues) {
    setFormError(null);
    try {
      await login(values);
      navigate(redirectTo ?? (await resolvePostLoginPath()), { replace: true });
    } catch (error) {
      setFormError(errorMessage(error, "Couldn't log you in. Check your email and password."));
    }
  }

  return (
    <div>
      <h1 className="text-lg font-semibold text-ink">Welcome back</h1>
      <p className="mt-1 text-sm text-ink-secondary">Log in to pick up your search where you left off.</p>

      <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4" noValidate>
        <div className="space-y-1.5">
          <label htmlFor="email" className="text-xs font-medium text-ink-secondary">
            Email
          </label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            invalid={!!errors.email}
            {...register("email")}
          />
          {errors.email && <p className="text-xs text-danger">{errors.email.message}</p>}
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label htmlFor="password" className="text-xs font-medium text-ink-secondary">
              Password
            </label>
            <Link to={paths.forgotPassword} className="text-xs font-medium text-brand hover:underline">
              Forgot password?
            </Link>
          </div>
          <PasswordInput
            id="password"
            autoComplete="current-password"
            invalid={!!errors.password}
            {...register("password")}
          />
          {errors.password && <p className="text-xs text-danger">{errors.password.message}</p>}
        </div>

        {formError && (
          <p role="alert" className="text-sm text-danger">
            {formError}
          </p>
        )}

        <Button type="submit" className="w-full" isLoading={isSubmitting}>
          Log in
          <ArrowRight className="size-4" aria-hidden />
        </Button>
      </form>

      <p className="mt-6 text-center text-sm text-ink-secondary">
        Don't have an account?{" "}
        <Link to={paths.register} className="font-medium text-brand hover:underline">
          Create one
        </Link>
      </p>
    </div>
  );
}
