let selectedExercise = "";
let videoFile = null;

const exerciseSelect = document.getElementById("exerciseSelect");
const videoInput = document.getElementById("videoInput");
const videoContainer = document.getElementById("videoContainer");
const output = document.getElementById("output");
const uploadBtn = document.getElementById("uploadBtn");
const dashboardBtn = document.getElementById("dashboardBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

// Dropdown selection
exerciseSelect.addEventListener("change", (e) => {
  selectedExercise = e.target.value;
});

// Dashboard button click → navigate to dashboard
dashboardBtn.addEventListener("click", () => {
  window.location.href = "/dashboard.html";
});

// Upload button click → open file picker
uploadBtn.addEventListener("click", () => {
  videoInput.click();
});

// Video selection → preview + upload to backend
videoInput.addEventListener("change", async (e) => {
  videoFile = e.target.files[0];
  if (!videoFile) return;

  // 1️⃣ Show preview in browser
  const videoUrl = URL.createObjectURL(videoFile);
  videoContainer.innerHTML = `<video src="${videoUrl}" controls></video>`;

  // 2️⃣ Send file to backend for saving
  const formData = new FormData();
  formData.append("file", videoFile);

  try {
    const res = await fetch("/api/save-video", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    output.textContent = `Video saved as: ${data.filename}`;
  } catch (err) {
    output.textContent = "Error saving video: " + err;
  }
});

// Analyze button click
analyzeBtn.addEventListener("click", async () => {
  if (!videoFile || !selectedExercise) {
    output.textContent = "Please select an exercise and upload a video first";
    return;
  }

  const formData = new FormData();
  formData.append("file", videoFile);
  formData.append("exercise", selectedExercise);

  try {
    const res = await fetch("/api/analyze-video", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    output.textContent = "Error analyzing video: " + err;
  }
});
