import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type View = "Dashboard" | "Cases" | "Evidence" | "Studio" | "Research" | "Settings";
type Case = { id: string; name: string; description: string; evidence_count: number; scan_count: number; created_at: string };
type Vault = { initialized: boolean; unlocked: boolean };

const views: { name: View; icon: string }[] = [
  { name: "Dashboard", icon: "⌁" }, { name: "Cases", icon: "▣" },
  { name: "Evidence", icon: "◇" }, { name: "Studio", icon: "◫" },
  { name: "Research", icon: "⌬" }, { name: "Settings", icon: "⚙" },
];

export function App() {
  const [view, setView] = useState<View>("Dashboard");
  const [csrf, setCsrf] = useState(sessionStorage.getItem("steg-csrf") || "");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [cases, setCases] = useState<Case[]>([]);
  const [vault, setVault] = useState<Vault>({ initialized: false, unlocked: false });
  const authenticated = Boolean(csrf);

  const request = async (path: string, options: RequestInit = {}) => {
    const headers = new Headers(options.headers);
    if (csrf && !["GET", "HEAD"].includes(options.method || "GET")) headers.set("X-CSRF-Token", csrf);
    const response = await fetch(path, { ...options, headers });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
    return response.status === 204 ? null : response.json();
  };

  const refresh = async () => {
    try {
      const [caseData, vaultData] = await Promise.all([request("/v2/cases"), request("/v2/vault")]);
      setCases(caseData.items); setVault(vaultData); setError("");
    } catch (reason) { setError(String(reason)); }
  };
  useEffect(() => { if (authenticated) void refresh(); }, [authenticated]);

  const login = async (event: FormEvent) => {
    event.preventDefault(); setError("");
    try {
      const response = await fetch("/v2/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: token }) });
      if (!response.ok) throw new Error("The local API token was not accepted.");
      const data = await response.json(); sessionStorage.setItem("steg-csrf", data.csrf_token); setCsrf(data.csrf_token); setToken("");
    } catch (reason) { setError(String(reason)); }
  };

  const stats = useMemo(() => ({ cases: cases.length, evidence: cases.reduce((n, item) => n + item.evidence_count, 0), scans: cases.reduce((n, item) => n + item.scan_count, 0) }), [cases]);
  if (!authenticated) return <Login token={token} setToken={setToken} login={login} error={error} />;

  return <div className="shell">
    <aside><div className="brand"><span className="brandMark">S</span><div><strong>steganography</strong><small>LOCAL DFIR PLATFORM</small></div></div>
      <nav>{views.map(item => <button className={view === item.name ? "active" : ""} onClick={() => setView(item.name)} key={item.name}><span>{item.icon}</span>{item.name}</button>)}</nav>
      <div className="local"><i></i><div><b>Local-only</b><small>No network egress</small></div></div>
    </aside>
    <main><header><div><p className="eyebrow">WORKSPACE / {view.toUpperCase()}</p><h1>{view}</h1></div><div className={`vault ${vault.unlocked ? "ok" : "locked"}`}>● Vault {vault.unlocked ? "unlocked" : "locked"}</div></header>
      {error && <div className="alert">{error}<button onClick={() => setError("")}>×</button></div>}
      {view === "Dashboard" && <Dashboard stats={stats} cases={cases} setView={setView} />}
      {view === "Cases" && <Cases cases={cases} request={request} refresh={refresh} />}
      {view === "Evidence" && <Evidence cases={cases} vault={vault} request={request} refresh={refresh} />}
      {view === "Studio" && <Studio request={request} />}
      {view === "Research" && <Research />}
      {view === "Settings" && <Settings vault={vault} request={request} refresh={refresh} />}
    </main>
  </div>;
}

function Login({ token, setToken, login, error }: { token: string; setToken: (v: string) => void; login: (e: FormEvent) => void; error: string }) {
  return <div className="login"><div className="loginGlow"></div><form onSubmit={login}><span className="brandMark large">S</span><p className="eyebrow">LOCAL-FIRST SECURITY</p><h1>Open your workspace</h1><p>Enter the API token stored in your local state directory. It never leaves this device.</p><label>Local API token<input type="password" value={token} onChange={e => setToken(e.target.value)} autoFocus required /></label>{error && <div className="alert">{error}</div>}<button className="primary">Authenticate →</button><small>HTTP-only session · SameSite cookie · CSRF protected</small></form></div>;
}

function Dashboard({ stats, cases, setView }: { stats: { cases: number; evidence: number; scans: number }; cases: Case[]; setView: (v: View) => void }) {
  return <><section className="hero"><div><p className="eyebrow">INVESTIGATION CONSOLE</p><h2>Find what the pixels<br/><em>aren't telling you.</em></h2><p>Case-oriented steganalysis, encrypted evidence custody and reproducible research — entirely on your machine.</p><button className="primary" onClick={() => setView("Cases")}>Start an investigation →</button></div><div className="signal"><span></span><span></span><span></span><span></span><b>BIT-PLANE<br/>SIGNAL MAP</b></div></section>
    <section className="metrics"><Metric label="OPEN CASES" value={stats.cases}/><Metric label="EVIDENCE FILES" value={stats.evidence}/><Metric label="ANALYSES" value={stats.scans}/><Metric label="NETWORK CALLS" value="0" accent/></section>
    <section className="panel"><div className="panelHead"><div><p className="eyebrow">RECENT ACTIVITY</p><h3>Cases</h3></div><button onClick={() => setView("Cases")}>View all</button></div>{cases.length ? cases.slice(0, 5).map(item => <div className="caseRow" key={item.id}><span className="caseIcon">▣</span><div><b>{item.name}</b><small>{item.evidence_count} evidence · {item.scan_count} scans</small></div><time>{new Date(item.created_at).toLocaleDateString()}</time></div>) : <Empty title="No cases yet" text="Create a case to begin preserving and analyzing evidence."/>}</section></>;
}
function Metric({label,value,accent=false}:{label:string;value:number|string;accent?:boolean}) { return <div><small>{label}</small><strong className={accent ? "accent" : ""}>{value}</strong></div>; }

function Cases({ cases, request, refresh }: { cases: Case[]; request: Function; refresh: () => Promise<void> }) {
  const [name,setName]=useState(""); const create=async(e:FormEvent)=>{e.preventDefault();await request("/v2/cases",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});setName("");await refresh();};
  return <div className="grid"><section className="panel"><div className="panelHead"><div><p className="eyebrow">CASE INDEX</p><h3>Investigations</h3></div></div>{cases.map(item=><div className="caseRow" key={item.id}><span className="caseIcon">▣</span><div><b>{item.name}</b><small>{item.description || "No description"}</small></div><code>{item.id.slice(0,8)}</code></div>)}{!cases.length&&<Empty title="No investigations" text="Your local case index is empty."/>}</section><form className="panel form" onSubmit={create}><p className="eyebrow">NEW CASE</p><h3>Create investigation</h3><label>Case name<input value={name} onChange={e=>setName(e.target.value)} placeholder="Operation Lighthouse" required/></label><button className="primary">Create case</button></form></div>;
}

function Evidence({ cases, vault, request, refresh }: { cases: Case[]; vault: Vault; request: Function; refresh: () => Promise<void> }) {
  const [caseId,setCaseId]=useState(cases[0]?.id||""); const [file,setFile]=useState<File|null>(null); const [busy,setBusy]=useState(false);
  const upload=async(e:FormEvent)=>{e.preventDefault();if(!file)return;setBusy(true);try{const body=new FormData();body.append("file",file);await request(`/v2/cases/${caseId}/evidence`,{method:"POST",body});await refresh();setFile(null);}finally{setBusy(false)}};
  const scan=async()=>{setBusy(true);try{await request(`/v2/cases/${caseId}/scans`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({profile:"sensitive"})});await refresh();}finally{setBusy(false)}};
  return <div className="grid"><form className="panel form" onSubmit={upload}><p className="eyebrow">CHAIN OF CUSTODY</p><h3>Add evidence</h3><label>Case<select value={caseId} onChange={e=>setCaseId(e.target.value)}>{cases.map(c=><option value={c.id} key={c.id}>{c.name}</option>)}</select></label><label className="drop">◇<b>{file?.name||"Choose evidence file"}</b><small>SHA-256 deduplicated · encrypted at rest</small><input type="file" onChange={e=>setFile(e.target.files?.[0]||null)}/></label><button className="primary" disabled={!vault.unlocked||!caseId||!file||busy}>{vault.unlocked?"Preserve evidence":"Unlock vault first"}</button></form><section className="panel form"><p className="eyebrow">BATCH ANALYSIS</p><h3>Scan a case</h3><p>Run every compatible deterministic, statistical and optional tool adapter against the preserved evidence.</p><button className="primary" onClick={scan} disabled={!vault.unlocked||!caseId||busy}>Queue full scan →</button></section></div>;
}

function Studio({ request }: { request: Function }) {
  const [carrier,setCarrier]=useState<File|null>(null);const [payload,setPayload]=useState<File|null>(null);const [result,setResult]=useState<any>(null);const run=async(kind:"capacity"|"embed"|"extract")=>{if(!carrier)return;const body=new FormData();body.append("carrier",carrier);if(payload)body.append("payload",payload);setResult(await request(`/v2/studio/${kind}`,{method:"POST",body}));};
  return <section className="panel studio"><p className="eyebrow">PRIVACY STUDIO</p><h3>Embed, inspect, recover</h3><div className="studioGrid"><label className="drop">◫<b>{carrier?.name||"Carrier file"}</b><input type="file" onChange={e=>setCarrier(e.target.files?.[0]||null)}/></label><label className="drop">◇<b>{payload?.name||"Payload file"}</b><input type="file" onChange={e=>setPayload(e.target.files?.[0]||null)}/></label></div><div className="actions"><button onClick={()=>run("capacity")}>Check capacity</button><button onClick={()=>run("extract")}>Extract</button><button className="primary" onClick={()=>run("embed")} disabled={!payload}>Embed payload</button></div>{result&&<pre>{JSON.stringify(result,null,2)}</pre>}</section>;
}
function Research(){return <section className="panel"><p className="eyebrow">REPRODUCIBLE RESEARCH</p><h3>Model laboratory</h3><div className="research"><div><b>Dataset isolation</b><p>Import user-supplied ALASKA2/BOSSBase-style directories with immutable SHA-256 manifests and seeded splits.</p></div><div><b>ONNX model domains</b><p><code>spatial-srnet-v1</code><br/><code>jpeg-srnet-v1</code></p></div><div><b>Evidence thresholds</b><p>Unsupported distributions remain inconclusive. Model availability never changes deterministic evidence.</p></div></div></section>}
function Settings({vault,request,refresh}:{vault:Vault;request:Function;refresh:()=>Promise<void>}){const[p,setP]=useState("");const act=async()=>{await request(vault.initialized?"/v2/vault/unlock":"/v2/vault/initialize",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:p})});setP("");await refresh()};return <div className="grid"><section className="panel form"><p className="eyebrow">ENCRYPTED VAULT</p><h3>{vault.unlocked?"Vault is unlocked":vault.initialized?"Unlock evidence vault":"Initialize evidence vault"}</h3><label>Vault password<input type="password" value={p} onChange={e=>setP(e.target.value)}/></label><button className="primary" onClick={act}>{vault.initialized?"Unlock":"Initialize"}</button>{vault.unlocked&&<button onClick={async()=>{await request("/v2/vault/lock",{method:"POST"});await refresh()}}>Lock now</button>}</section><section className="panel"><p className="eyebrow">SECURITY POSTURE</p><h3>Local by design</h3><ul><li>Argon2id master key derivation</li><li>libsodium XChaCha20 secretstream</li><li>HTTP-only SameSite sessions</li><li>Network access disabled by default</li></ul></section></div>}
function Empty({title,text}:{title:string;text:string}){return <div className="empty"><span>⌁</span><b>{title}</b><p>{text}</p></div>}

const rootElement = document.getElementById("root");
if (rootElement) createRoot(rootElement).render(<React.StrictMode><App /></React.StrictMode>);
