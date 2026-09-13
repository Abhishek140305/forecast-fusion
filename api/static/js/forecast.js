async function loadForecast() {
  const response = await fetch("/forecast");
  const data = await response.json();
  const prediction = data.prediction;
  const actual = data.actual;

  const labels = prediction.map((_, i) => `Hour ${i + 1}`);

  const ctx = document.getElementById("forecastChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Predicted Temperature (°C)",
          data: prediction,
          borderColor: "rgba(0, 200, 255, 1)",
          backgroundColor: "rgba(0, 200, 255, 0.2)",
          fill: true,
          tension: 0.3,
        },
        {
          label: "Actual Temperature (°C)",
          data: actual,
          borderColor: "rgba(255, 99, 132, 1)",
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: "Real vs Predicted (Next 72 Hours)",
        },
      },
    },
  });
}
