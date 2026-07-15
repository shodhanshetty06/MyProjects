import { Navigate, Route, Routes } from "react-router-dom";

import { MainLayout } from "@/layouts/MainLayout";

import {
  AdminLayout,
  ApplicationsPage as AdminApplicationsPage,
  AuditLogsPage,
  CompaniesPage,
  DashboardPage as AdminDashboardPage,
  FeatureFlagsPage,
  JobsPage,
  ModerationQueuePage,
  PermissionsPage,
  PlatformAnalyticsPage,
  ReportsPage,
  SystemLogsPage,
  UserDetailPage,
  UsersPage,
} from "@/features/admin";
import { AICareerCoachPage } from "@/features/ai-career-coach";
import { AnalyticsPage } from "@/features/analytics";
import { ATSResumeGeneratorPage } from "@/features/ats-resume-generator";
import { AuthPage, ProtectedRoute } from "@/features/auth";
import { AutoApplyPage } from "@/features/auto-apply";
import { JobIngestionPage } from "@/features/job-ingestion";
import { MatchingPage } from "@/features/matching";
import { NotificationPage } from "@/features/notification";
import { ResumeParserPage } from "@/features/resume-parser";
import { SearchPage } from "@/features/search";
import { TrackingPage } from "@/features/tracking";
import { UserProfilePage } from "@/features/user-profile";

/**
 * Top-level route table. Most modules still render one placeholder page
 * proving the frontend can reach their API scaffold; `/admin` is fully
 * implemented (Phase 12) with its own nested route tree under `AdminLayout`.
 */
export function AppRoutes() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/auth" element={<AuthPage />} />
        <Route element={<ProtectedRoute />}>
          <Route index element={<Navigate to="/search" replace />} />
          <Route path="/user-profile" element={<UserProfilePage />} />
          <Route path="/resume-parser" element={<ResumeParserPage />} />
          <Route path="/job-ingestion" element={<JobIngestionPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/matching" element={<MatchingPage />} />
          <Route path="/ats-resume-generator" element={<ATSResumeGeneratorPage />} />
          <Route path="/auto-apply" element={<AutoApplyPage />} />
          <Route path="/tracking" element={<TrackingPage />} />
          <Route path="/notification" element={<NotificationPage />} />
          <Route path="/ai-career-coach" element={<AICareerCoachPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<AdminDashboardPage />} />
            <Route path="users" element={<UsersPage />} />
            <Route path="users/:userId" element={<UserDetailPage />} />
            <Route path="companies" element={<CompaniesPage />} />
            <Route path="jobs" element={<JobsPage />} />
            <Route path="applications" element={<AdminApplicationsPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="moderation" element={<ModerationQueuePage />} />
            <Route path="analytics" element={<PlatformAnalyticsPage />} />
            <Route path="system-logs" element={<SystemLogsPage />} />
            <Route path="audit-logs" element={<AuditLogsPage />} />
            <Route path="feature-flags" element={<FeatureFlagsPage />} />
            <Route path="permissions" element={<PermissionsPage />} />
          </Route>
        </Route>
      </Route>
    </Routes>
  );
}
