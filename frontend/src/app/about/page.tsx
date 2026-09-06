export default function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto space-y-8 bg-white p-8 rounded-lg shadow-sm border border-slate-200">
      <header className="border-b pb-6">
        <h1 className="text-3xl font-bold text-slate-800">About AgriFL</h1>
        <p className="text-slate-600 mt-2 text-lg">
          Privacy-Preserving Intelligent Crop Recommendation using Federated Learning
        </p>
      </header>

      <section className="space-y-4 text-slate-700 leading-relaxed">
        <p>
          AgriFL is a final year project that uses agricultural parameters to recommend suitable crops while exploring privacy-preserving Federated Learning.
        </p>
        
        <h2 className="text-xl font-semibold text-slate-800 pt-4">System Architecture</h2>
        <div className="bg-slate-50 p-6 rounded-md font-mono text-sm border border-slate-200">
          <pre>{`Agricultural Data
       ↓
Distributed Clients
       ↓
Local Model Training
       ↓
FedAvg
       ↓
Federated Global Model
       ↓
Crop Recommendation`}</pre>
        </div>

        <h2 className="text-xl font-semibold text-slate-800 pt-4">Model Strategy</h2>
        <p>This system has multiple model implementations, clearly distinguishing between centralized research baselines and the final federated application model.</p>
        
        <ul className="space-y-3 list-disc list-inside bg-emerald-50 p-6 rounded-md border border-emerald-100">
          <li>
            <strong>Centralized baseline:</strong> Random Forest
          </li>
          <li>
            <strong>Centralized deep learning:</strong> DNN, CNN-LSTM
          </li>
          <li>
            <strong>Final application model:</strong> <code>fl_global_model.keras</code>
          </li>
        </ul>
        
        <p className="mt-4">
          This frontend interface strictly uses the <strong>Final application model</strong> (<code>fl_global_model.keras</code>) via the backend API.
        </p>
      </section>
    </div>
  );
}
