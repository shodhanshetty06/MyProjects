import type { Transition, Variants } from "framer-motion";

export const easeOut: Transition["ease"] = [0.16, 1, 0.3, 1];

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.35, ease: easeOut } },
};

export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: easeOut } },
};

export const fadeInScale: Variants = {
  hidden: { opacity: 0, scale: 0.97 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.3, ease: easeOut } },
};

export const staggerChildren = (stagger = 0.06, delayChildren = 0): Variants => ({
  hidden: {},
  visible: {
    transition: { staggerChildren: stagger, delayChildren },
  },
});

export const slideInFromRight: Variants = {
  hidden: { opacity: 0, x: 24 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.35, ease: easeOut } },
  exit: { opacity: 0, x: 24, transition: { duration: 0.2, ease: easeOut } },
};

export const collapseHeight: Variants = {
  hidden: { height: 0, opacity: 0 },
  visible: { height: "auto", opacity: 1, transition: { duration: 0.25, ease: easeOut } },
  exit: { height: 0, opacity: 0, transition: { duration: 0.2, ease: easeOut } },
};

export const pageTransition: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: easeOut } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2, ease: easeOut } },
};
