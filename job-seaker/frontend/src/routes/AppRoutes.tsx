import { lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { AuthShell } from "@/components/layout/AuthShell";
import { MarketingShell } from "@/components/layout/MarketingShell";
import { ProtectedRoute } from "@/features/auth";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { paths } from "@/routes/paths";

/**
 * Every route-level page is lazy-loaded from its own file (bypassing
 * feature barrels, which re-export sibling pages too and would otherwise
 * collapse the split back to one chunk per feature) so each route only
 * ships the JS it needs. Shells/layouts stay eager since they render
 * immediately on every navigation.
 */
const LandingPage = lazy(() => import("@/features/marketing/pages/LandingPage").then((m) => ({ default: m.LandingPage })));
const FeaturesPage = lazy(() => import("@/features/marketing/pages/FeaturesPage").then((m) => ({ default: m.FeaturesPage })));
const PricingPage = lazy(() => import("@/features/marketing/pages/PricingPage").then((m) => ({ default: m.PricingPage })));
const AboutPage = lazy(() => import("@/features/marketing/pages/AboutPage").then((m) => ({ default: m.AboutPage })));
const ContactPage = lazy(() => import("@/features/marketing/pages/ContactPage").then((m) => ({ default: m.ContactPage })));

const LoginPage = lazy(() => import("@/features/auth/pages/LoginPage").then((m) => ({ default: m.LoginPage })));
const RegisterPage = lazy(() => import("@/features/auth/pages/RegisterPage").then((m) => ({ default: m.RegisterPage })));
const ForgotPasswordPage = lazy(() => import("@/features/auth/pages/ForgotPasswordPage").then((m) => ({ default: m.ForgotPasswordPage })));
const ResetPasswordPage = lazy(() => import("@/features/auth/pages/ResetPasswordPage").then((m) => ({ default: m.ResetPasswordPage })));
const VerifyEmailPage = lazy(() => import("@/features/auth/pages/VerifyEmailPage").then((m) => ({ default: m.VerifyEmailPage })));

const WelcomePage = lazy(() => import("@/features/onboarding/pages/WelcomePage").then((m) => ({ default: m.WelcomePage })));
const ProfileReviewPage = lazy(() => import("@/features/onboarding/pages/ProfileReviewPage").then((m) => ({ default: m.ProfileReviewPage })));
const WorkspacePage = lazy(() => import("@/features/workspace/pages/WorkspacePage").then((m) => ({ default: m.WorkspacePage })));

const ResumeManagerPage = lazy(() => import("@/features/resume-parser/pages/ResumeManagerPage").then((m) => ({ default: m.ResumeManagerPage })));
const ResumeDetailPage = lazy(() => import("@/features/resume-parser/pages/ResumeDetailPage").then((m) => ({ default: m.ResumeDetailPage })));
const ResumeVersionViewerPage = lazy(() => import("@/features/resume-parser/pages/ResumeVersionViewerPage").then((m) => ({ default: m.ResumeVersionViewerPage })));
const ATSResumeGeneratorPage = lazy(() => import("@/features/ats-resume-generator/pages/ATSResumeGeneratorPage").then((m) => ({ default: m.ATSResumeGeneratorPage })));

const JobSearchHomePage = lazy(() => import("@/features/search/pages/JobSearchHomePage").then((m) => ({ default: m.JobSearchHomePage })));
const JobDetailPage = lazy(() => import("@/features/search/pages/JobDetailPage").then((m) => ({ default: m.JobDetailPage })));
const SavedJobsPage = lazy(() => import("@/features/search/pages/SavedJobsPage").then((m) => ({ default: m.SavedJobsPage })));
const CompanyExplorerPage = lazy(() => import("@/features/job-ingestion/pages/CompanyExplorerPage").then((m) => ({ default: m.CompanyExplorerPage })));
const CompanyDetailPage = lazy(() => import("@/features/job-ingestion/pages/CompanyDetailPage").then((m) => ({ default: m.CompanyDetailPage })));

const MatchingPage = lazy(() => import("@/features/matching/pages/MatchingPage").then((m) => ({ default: m.MatchingPage })));

const AutoApplyPage = lazy(() => import("@/features/auto-apply/pages/AutoApplyPage").then((m) => ({ default: m.AutoApplyPage })));
const AutoApplyTaskDetailPage = lazy(() => import("@/features/auto-apply/pages/AutoApplyTaskDetailPage").then((m) => ({ default: m.AutoApplyTaskDetailPage })));

const TrackerBoardPage = lazy(() => import("@/features/tracking/pages/TrackerBoardPage").then((m) => ({ default: m.TrackerBoardPage })));
const TrackerCalendarPage = lazy(() => import("@/features/tracking/pages/TrackerCalendarPage").then((m) => ({ default: m.TrackerCalendarPage })));
const TrackerEmailsPage = lazy(() => import("@/features/tracking/pages/TrackerEmailsPage").then((m) => ({ default: m.TrackerEmailsPage })));
const TrackerTimelinePage = lazy(() => import("@/features/tracking/pages/TrackerTimelinePage").then((m) => ({ default: m.TrackerTimelinePage })));

const NotificationPage = lazy(() => import("@/features/notification/pages/NotificationPage").then((m) => ({ default: m.NotificationPage })));

const CoachHomePage = lazy(() => import("@/features/ai-career-coach/pages/CoachHomePage").then((m) => ({ default: m.CoachHomePage })));
const CoachChatPage = lazy(() => import("@/features/ai-career-coach/pages/CoachChatPage").then((m) => ({ default: m.CoachChatPage })));
const CoachInsightsPage = lazy(() => import("@/features/ai-career-coach/pages/CoachInsightsPage").then((m) => ({ default: m.CoachInsightsPage })));
const CoachToolPage = lazy(() => import("@/features/ai-career-coach/pages/CoachToolPage").then((m) => ({ default: m.CoachToolPage })));

const AnalyticsPage = lazy(() => import("@/features/analytics/pages/AnalyticsPage").then((m) => ({ default: m.AnalyticsPage })));

const SettingsProfilePage = lazy(() => import("@/features/settings/pages/SettingsProfilePage").then((m) => ({ default: m.SettingsProfilePage })));
const SettingsSecurityPage = lazy(() => import("@/features/settings/pages/SettingsSecurityPage").then((m) => ({ default: m.SettingsSecurityPage })));
const SettingsApiKeysPage = lazy(() => import("@/features/settings/pages/SettingsApiKeysPage").then((m) => ({ default: m.SettingsApiKeysPage })));

const AdminLayout = lazy(() => import("@/features/admin/AdminLayout").then((m) => ({ default: m.AdminLayout })));
const AdminDashboardPage = lazy(() => import("@/features/admin/pages/DashboardPage").then((m) => ({ default: m.DashboardPage })));
const AdminUsersPage = lazy(() => import("@/features/admin/pages/UsersPage").then((m) => ({ default: m.UsersPage })));
const AdminUserDetailPage = lazy(() => import("@/features/admin/pages/UserDetailPage").then((m) => ({ default: m.UserDetailPage })));
const AdminCompaniesPage = lazy(() => import("@/features/admin/pages/CompaniesPage").then((m) => ({ default: m.CompaniesPage })));
const AdminJobsPage = lazy(() => import("@/features/admin/pages/JobsPage").then((m) => ({ default: m.JobsPage })));
const AdminApplicationsPage = lazy(() => import("@/features/admin/pages/ApplicationsPage").then((m) => ({ default: m.ApplicationsPage })));
const AdminReportsPage = lazy(() => import("@/features/admin/pages/ReportsPage").then((m) => ({ default: m.ReportsPage })));
const AdminModerationQueuePage = lazy(() => import("@/features/admin/pages/ModerationQueuePage").then((m) => ({ default: m.ModerationQueuePage })));
const AdminPlatformAnalyticsPage = lazy(() => import("@/features/admin/pages/PlatformAnalyticsPage").then((m) => ({ default: m.PlatformAnalyticsPage })));
const AdminSystemLogsPage = lazy(() => import("@/features/admin/pages/SystemLogsPage").then((m) => ({ default: m.SystemLogsPage })));
const AdminAuditLogsPage = lazy(() => import("@/features/admin/pages/AuditLogsPage").then((m) => ({ default: m.AuditLogsPage })));
const AdminFeatureFlagsPage = lazy(() => import("@/features/admin/pages/FeatureFlagsPage").then((m) => ({ default: m.FeatureFlagsPage })));
const AdminPermissionsPage = lazy(() => import("@/features/admin/pages/PermissionsPage").then((m) => ({ default: m.PermissionsPage })));

/** Top-level route table. `/admin` has its own nested route tree under `AdminLayout`. */
export function AppRoutes() {
  return (
    <Routes>
        <Route element={<MarketingShell />}>
          <Route path={paths.home} element={<LandingPage />} />
          <Route path={paths.features} element={<FeaturesPage />} />
          <Route path={paths.pricing} element={<PricingPage />} />
          <Route path={paths.about} element={<AboutPage />} />
          <Route path={paths.contact} element={<ContactPage />} />
        </Route>

        <Route element={<AuthShell />}>
          <Route path={paths.login} element={<LoginPage />} />
          <Route path={paths.register} element={<RegisterPage />} />
          <Route path={paths.forgotPassword} element={<ForgotPasswordPage />} />
          <Route path={paths.resetPassword} element={<ResetPasswordPage />} />
          <Route path={paths.verifyEmail} element={<VerifyEmailPage />} />
        </Route>

        <Route element={<ProtectedRoute />}>
          {/* Bare (no AppShell sidebar) — a first-time user has nothing in
              the nav yet to make sense of. */}
          <Route path={paths.onboarding} element={<WelcomePage />} />
          <Route path={paths.onboardingReview()} element={<ProfileReviewPage />} />

          <Route element={<AppShell />}>
            <Route path={paths.dashboard} element={<Navigate to={paths.workspace} replace />} />
            <Route path={paths.workspace} element={<WorkspacePage />} />

            <Route path={paths.resumes} element={<ResumeManagerPage />} />
            <Route path={paths.resumeDetail()} element={<ResumeDetailPage />} />
            <Route path={paths.resumeVersion()} element={<ResumeVersionViewerPage />} />
            <Route path={paths.atsGenerator} element={<ATSResumeGeneratorPage />} />

            <Route path={paths.search} element={<JobSearchHomePage />} />
            <Route path={paths.jobDetail()} element={<JobDetailPage />} />
            <Route path={paths.savedJobs} element={<SavedJobsPage />} />
            <Route path={paths.companies} element={<CompanyExplorerPage />} />
            <Route path={paths.companyDetail()} element={<CompanyDetailPage />} />

            <Route path="/matching" element={<MatchingPage />} />

            <Route path={paths.autoApply} element={<AutoApplyPage />} />
            <Route path={paths.autoApplyTask()} element={<AutoApplyTaskDetailPage />} />

            <Route path={paths.tracker} element={<TrackerBoardPage />} />
            <Route path={paths.trackerCalendar} element={<TrackerCalendarPage />} />
            <Route path={paths.trackerEmails} element={<TrackerEmailsPage />} />
            <Route path={paths.trackerTimeline()} element={<TrackerTimelinePage />} />

            <Route path={paths.notifications} element={<NotificationPage />} />

            <Route path={paths.coach} element={<CoachHomePage />} />
            <Route path={paths.coachChat} element={<CoachChatPage />} />
            <Route path={paths.coachInsights} element={<CoachInsightsPage />} />
            <Route path={paths.coachTool()} element={<CoachToolPage />} />

            <Route path={paths.analytics} element={<AnalyticsPage />} />

            <Route path={paths.settings} element={<Navigate to={paths.settingsProfile} replace />} />
            <Route path={paths.settingsProfile} element={<SettingsProfilePage />} />
            <Route path={paths.settingsSecurity} element={<SettingsSecurityPage />} />
            <Route path={paths.settingsApiKeys} element={<SettingsApiKeysPage />} />

            <Route path="/admin" element={<AdminLayout />}>
              <Route index element={<AdminDashboardPage />} />
              <Route path="users" element={<AdminUsersPage />} />
              <Route path="users/:userId" element={<AdminUserDetailPage />} />
              <Route path="companies" element={<AdminCompaniesPage />} />
              <Route path="jobs" element={<AdminJobsPage />} />
              <Route path="applications" element={<AdminApplicationsPage />} />
              <Route path="reports" element={<AdminReportsPage />} />
              <Route path="moderation" element={<AdminModerationQueuePage />} />
              <Route path="analytics" element={<AdminPlatformAnalyticsPage />} />
              <Route path="system-logs" element={<AdminSystemLogsPage />} />
              <Route path="audit-logs" element={<AdminAuditLogsPage />} />
              <Route path="feature-flags" element={<AdminFeatureFlagsPage />} />
              <Route path="permissions" element={<AdminPermissionsPage />} />
            </Route>
          </Route>
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
  );
}
