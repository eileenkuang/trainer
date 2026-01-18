const videoElement = document.getElementById("dashboardVideo");

// Load last uploaded video
const filename = localStorage.getItem("lastVideo");

if (filename) {
  videoElement.src = `/uploads/${filename}`;
} else {
  videoElement.style.display = "none";
}
