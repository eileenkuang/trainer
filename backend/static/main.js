async function ping() {
  const res = await fetch("/api/ping");
  const data = await res.json();
  console.log(data);
}

document.getElementById("pingBtn").onclick = async () => {
  const res = await fetch("http://localhost:8000/api/ping");
  const data = await res.json();
  document.getElementById("output").textContent =
    JSON.stringify(data, null, 2);
};
