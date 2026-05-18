function login() {
    if (user.value === "admin" && pass.value === "1234") {
        window.location.href = "dashboard.html";
    } else alert("Wrong credentials");
}

function toggleTheme() {
    document.body.classList.toggle("light");
}

let historyData = JSON.parse(localStorage.getItem("history")) || [];

async function checkFraud() {

    let data = {
        amount: parseFloat(amount.value),
        spending_deviation_score: parseFloat(deviation.value),
        velocity_score: parseFloat(velocity.value),
        geo_anomaly_score: parseFloat(geo.value)
    };

    let res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    let result = await res.json();
    let risk = result.risk_level;

    resultDiv = document.getElementById("result");
    resultDiv.innerHTML = `<div class="${risk.toLowerCase()}">Risk: ${risk}</div>`;

    historyData.push({amount: data.amount, risk: risk});
    localStorage.setItem("history", JSON.stringify(historyData));

    updateKPIs();
    updateLiveFeed();
}

function updateKPIs() {
    if (!totalTx) return;

    totalTx.innerText = historyData.length;
    fraudCount.innerText = historyData.filter(r => r.risk === "HIGH").length;
    riskStatus.innerText = historyData.length ? historyData[historyData.length-1].risk : "-";
}

function updateLiveFeed() {
    if (!liveFeed) return;

    liveFeed.innerHTML = "";
    historyData.slice(-5).reverse().forEach(r => {
        liveFeed.innerHTML += `<div>₹${r.amount} → ${r.risk}</div>`;
    });
}

function loadChart() {
    if (!chart) return;

    let counts = {LOW:0, MEDIUM:0, HIGH:0};
    historyData.forEach(r => counts[r.risk]++);

    new Chart(chart, {
        type: "doughnut",
        data: {
            labels: ["LOW", "MEDIUM", "HIGH"],
            datasets: [{
                data: [counts.LOW, counts.MEDIUM, counts.HIGH],
                backgroundColor: ["green", "orange", "red"]
            }]
        },
        options: {
            animation: { animateRotate: true }
        }
    });
}

if (document.getElementById("chart")) loadChart();

function updateHistory() {
    if (!historyTable) return;

    historyTable.innerHTML = "";
    historyData.forEach(r => {
        historyTable.innerHTML += `<tr><td>${r.amount}</td><td>${r.risk}</td></tr>`;
    });
}

updateHistory();

function downloadCSV() {
    let csv = "Amount,Risk\n";
    historyData.forEach(r => csv += `${r.amount},${r.risk}\n`);

    let blob = new Blob([csv]);
    let link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "report.csv";
    link.click();
}

setInterval(updateKPIs, 2000);
setInterval(updateLiveFeed, 2000);