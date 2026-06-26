import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { AdversarialArtifact, Dataset, Explanation, Flag, Metrics, StreamStats, SummaryStatus } from '../models/api.models';

/** Calls the Spring Boot BFF (which orchestrates the FastAPI inference service). */
@Injectable({ providedIn: 'root' })
export class ApiService {
  // Override at build time / via a proxy; defaults to the local Spring Boot port.
  private readonly base = (window as any).AEGIS_API ?? 'http://localhost:8080/api';

  constructor(private http: HttpClient) {}

  datasets(): Observable<Dataset[]> { return this.http.get<Dataset[]>(`${this.base}/datasets`); }

  flags(datasetId: number, threshold = 0.5, limit = 100): Observable<Flag[]> {
    return this.http.get<Flag[]>(`${this.base}/flags/${datasetId}`, { params: { threshold, limit } as any });
  }

  metrics(datasetId: number, split = 'test'): Observable<Metrics> {
    return this.http.get<Metrics>(`${this.base}/metrics/${datasetId}`, { params: { split } });
  }

  explain(datasetId: number, nodeId: number): Observable<Explanation> {
    return this.http.get<Explanation>(`${this.base}/explain/${datasetId}/${nodeId}`);
  }

  /** Explanation + (best-effort, inline) AI summary in one round-trip; the backend fans the two out
   *  concurrently and returns the summary if it lands quickly, else summary_pending stays true. */
  investigate(datasetId: number, nodeId: number): Observable<Explanation> {
    return this.http.get<Explanation>(`${this.base}/investigate/${datasetId}/${nodeId}`);
  }

  summary(datasetId: number, nodeId: number): Observable<SummaryStatus> {
    return this.http.get<SummaryStatus>(`${this.base}/summary/${datasetId}/${nodeId}`);
  }

  adversarial(): Observable<AdversarialArtifact> {
    return this.http.post<AdversarialArtifact>(`${this.base}/adversarial/run`, {});
  }

  /** SSE endpoint URL for the live transaction stream (consumed via EventSource). */
  streamUrl(): string { return `${this.base}/stream`; }

  streamStats(): Observable<StreamStats> {
    return this.http.get<StreamStats>(`${this.base}/stream/stats`);
  }
}
