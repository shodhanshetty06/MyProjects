/**
 * Central route path registry. Every route the app can navigate to is
 * named here so pages, the sidebar, and the command palette never
 * hand-write a path string more than once.
 */
export const paths = {
  // marketing (public)
  home: "/",
  features: "/features",
  pricing: "/pricing",
  about: "/about",
  contact: "/contact",

  // auth (public)
  login: "/login",
  register: "/register",
  forgotPassword: "/forgot-password",
  resetPassword: "/reset-password",
  verifyEmail: "/verify-email",

  // authenticated app
  dashboard: "/dashboard",

  search: "/jobs",
  jobDetail: (jobId = ":jobId") => `/jobs/${jobId}`,
  savedJobs: "/jobs/saved",
  recommendedJobs: "/jobs/recommended",
  companies: "/companies",
  companyDetail: (companyId = ":companyId") => `/companies/${companyId}`,

  resumes: "/resumes",
  resumeDetail: (resumeId = ":resumeId") => `/resumes/${resumeId}`,
  resumeVersion: (resumeId = ":resumeId", versionId = ":versionId") =>
    `/resumes/${resumeId}/versions/${versionId}`,
  atsGenerator: "/resumes/ats-generator",
  atsResumeDetail: (atsResumeId = ":atsResumeId") => `/resumes/ats-generator/${atsResumeId}`,
  resumeCompare: (resumeId = ":resumeId") => `/resumes/${resumeId}/compare`,

  autoApply: "/auto-apply",
  autoApplyTask: (taskId = ":taskId") => `/auto-apply/${taskId}`,

  tracker: "/tracker",
  trackerTimeline: (applicationId = ":applicationId") => `/tracker/${applicationId}`,
  trackerCalendar: "/tracker/calendar",
  trackerEmails: "/tracker/emails",

  notifications: "/notifications",

  coach: "/coach",
  coachChat: "/coach/chat",
  coachInsights: "/coach/insights",
  coachTool: (toolId = ":toolId") => `/coach/tools/${toolId}`,
  coachSkillGap: "/coach/tools/skill-gap",
  coachRoadmap: "/coach/tools/roadmap",
  coachSalary: "/coach/tools/salary-prediction",
  coachLearning: "/coach/tools/learning-recommendations",

  analytics: "/analytics",

  settings: "/settings",
  settingsProfile: "/settings/profile",
  settingsSecurity: "/settings/security",
  settingsApiKeys: "/settings/api-keys",

  admin: "/admin",

  notFound: "/404",
  serverError: "/500",
} as const;
