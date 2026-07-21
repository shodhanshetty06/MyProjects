/**
 * Job Ingestion feature - type contracts, mirroring
 * backend/app/modules/job_ingestion/schemas.py. These endpoints are
 * public (no auth header) - this module reads the company/job directory
 * that "Company Explorer" and job detail pages browse.
 */
export type WorkMode = "remote" | "hybrid" | "onsite" | "unknown";
export type EmploymentType =
  | "full_time"
  | "part_time"
  | "contract"
  | "internship"
  | "temporary"
  | "freelance"
  | "unknown";

export interface CompanyIntelligenceRead {
  ats_platform: string | null;
  ats_identifier: string | null;
  career_url: string | null;
  adapter_config: Record<string, unknown>;
  last_crawled_at: string | null;
  last_crawl_status: string | null;
  hiring_frequency_per_month: number | null;
  company_rating: number | null;
  hiring_trends: Record<string, unknown> | null;
}

export interface CompanyRead {
  id: string;
  name: string;
  normalized_name: string;
  website: string | null;
  logo_url: string | null;
  industry: string | null;
  description: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  intelligence: CompanyIntelligenceRead | null;
}

export interface LocationRead {
  id: string;
  raw_text: string;
  city: string | null;
  region: string | null;
  country: string | null;
  country_code: string | null;
}

export interface SkillRead {
  id: string;
  name: string;
}

export interface JobRead {
  id: string;
  company_id: string;
  location: LocationRead | null;
  source: string;
  source_url: string;
  title: string;
  description: string;
  employment_type: EmploymentType;
  work_mode: WorkMode;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  posted_at: string | null;
  status: string;
  skills: SkillRead[];
  has_embedding: boolean;
  created_at: string;
  updated_at: string;
}

export interface JobListFiltersParams {
  company_id?: string;
  status?: string;
  page?: number;
  page_size?: number;
}
