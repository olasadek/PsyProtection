# 🧠 PsyProtection: External Endpoint (EEP)

The **EEP** (External Endpoint) is the main orchestrator of the PsyProtection platform. It acts as a unified entry point where psychiatrists or hospital staff can submit patient data (MRI + EHR) and receive an end-to-end risk analysis powered by internal AI agents.

---

## 🚀 What Does EEP Do?

Upon receiving a request, the EEP:

1. **Validates the patient's MRI scan**
2. **Predicts drug abuse risk** using multimodal AI
3. **Generates a heatmap** (Grad-CAM) to explain the MRI-based prediction
4. **Fetches latest medical research** related to the risk
5. **Returns a complete report** with results, explanation, and treatment suggestions

---

## 📬 Endpoint

### `POST /analyze_patient`

#### 🔸 Form Data:
- `file`: Patient MRI scan (`.nii` or `.nii.gz`)
- `EHR_features`: JSON string of structured EHR fields
- `patient_id`: (Optional) ID or reference for patient

#### ✅ Example cURL:

```bash
curl -X POST http://localhost:9000/analyze_patient \
  -F "file=@mri_sample.nii.gz" \
  -F "EHR_features={\"age\":40}" \
  -F "patient_id=12345"
