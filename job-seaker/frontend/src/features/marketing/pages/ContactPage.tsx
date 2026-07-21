import { zodResolver } from "@hookform/resolvers/zod";
import { motion } from "framer-motion";
import { Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { fadeInUp } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { Textarea } from "@/components/ui/Textarea";
import { GradientBlob } from "@/features/marketing/components/GradientBlob";
import { Section } from "@/features/marketing/components/Section";

const contactSchema = z.object({
  name: z.string().min(1, "Enter your name"),
  email: z.string().email("Enter a valid email"),
  message: z.string().min(10, "Message should be at least 10 characters"),
});

type ContactForm = z.infer<typeof contactSchema>;

export function ContactPage() {
  const [submitted, setSubmitted] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<ContactForm>({ resolver: zodResolver(contactSchema) });

  async function onSubmit() {
    await new Promise((resolve) => setTimeout(resolve, 400));
    setSubmitted(true);
    reset();
  }

  return (
    <div>
      <div className="relative overflow-hidden">
        <GradientBlob className="-top-24 right-1/3 h-72 w-[36rem]" />
        <Section
          eyebrow="Contact"
          title="Get in touch"
          description="Questions, feedback, or something broken? Send a message and we'll follow up."
          className="pb-8"
        />
      </div>

      <Section className="pt-0" align="left">
        <motion.form
          variants={fadeInUp}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          onSubmit={handleSubmit(onSubmit)}
          className="mx-auto max-w-lg space-y-4 rounded-[var(--radius-2xl)] border border-border bg-surface p-7"
        >
          {submitted && (
            <NotConnectedNotice description="This environment doesn't have a message inbox wired up yet, so nothing was sent - but the form itself works end to end." />
          )}

          <div className="space-y-1.5">
            <label htmlFor="name" className="text-xs font-medium text-ink-secondary">
              Name
            </label>
            <Input id="name" placeholder="Ada Lovelace" invalid={!!errors.name} {...register("name")} />
            {errors.name && <p className="text-xs text-danger">{errors.name.message}</p>}
          </div>

          <div className="space-y-1.5">
            <label htmlFor="email" className="text-xs font-medium text-ink-secondary">
              Email
            </label>
            <Input id="email" type="email" placeholder="you@example.com" invalid={!!errors.email} {...register("email")} />
            {errors.email && <p className="text-xs text-danger">{errors.email.message}</p>}
          </div>

          <div className="space-y-1.5">
            <label htmlFor="message" className="text-xs font-medium text-ink-secondary">
              Message
            </label>
            <Textarea id="message" rows={5} placeholder="How can we help?" invalid={!!errors.message} {...register("message")} />
            {errors.message && <p className="text-xs text-danger">{errors.message.message}</p>}
          </div>

          <Button type="submit" className="w-full" isLoading={isSubmitting}>
            <Send className="size-4" aria-hidden />
            Send message
          </Button>
        </motion.form>
      </Section>
    </div>
  );
}
