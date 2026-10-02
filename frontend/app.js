const API_BASE =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:5000/api"
        : "/api";


/* ================= NAVIGATION ================= */

function showSection(sectionId, button) {

    document.querySelectorAll(".section").forEach(section => {
        section.classList.remove("active-section");
    });

    const section = document.getElementById(sectionId);

    if (section) {
        section.classList.add("active-section");
    }

    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.remove("active");
    });

    if (button) {
        button.classList.add("active");
    }

    const titles = {
        dashboard: "Analytics Dashboard",
        analysis: "Data Analysis",
        model: "ML Model Performance",
        prediction: "Heart Disease Prediction",
        about: "Project Methodology"
    };

    document.getElementById("pageTitle").textContent =
        titles[sectionId] || "CardioAI";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


/* ================= API ================= */

async function fetchAPI(endpoint) {

    const response = await fetch(`${API_BASE}${endpoint}`);

    if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
    }

    return await response.json();
}


/* ================= LOAD MODEL ================= */

async function loadModel() {

    try {

        const data = await fetchAPI("/model");

        const accuracy = `${data.accuracy}%`;

        document.getElementById("modelAccuracy").textContent = accuracy;
        document.getElementById("modelAccuracy2").textContent = accuracy;

        document.getElementById("modelPrecision").textContent =
            `${data.precision}%`;

        document.getElementById("modelRecall").textContent =
            `${data.recall}%`;

        document.getElementById("modelF1").textContent =
            `${data.f1_score}%`;

        document.getElementById("trainingRecords").textContent =
            data.training_records;

        document.getElementById("testingRecords").textContent =
            data.testing_records;

        document.getElementById("testingRecords2").textContent =
            data.testing_records;

        document.getElementById("totalRecords").textContent =
            data.total_records;

        document.getElementById("totalRecords2").textContent =
            data.total_records;

        document.getElementById("totalFeatures").textContent =
            data.features;

        document.getElementById("totalFeatures2").textContent =
            data.features;

        document.getElementById("heroRecords").textContent =
            data.total_records;

        document.getElementById("heroFeatures").textContent =
            data.features;

        renderConfusionMatrix(data.confusion_matrix);

        renderFeatureImportance(data.feature_importance);

    } catch (error) {

        console.error("Model API Error:", error);

    }
}


/* ================= EDA ================= */

async function loadEDA() {

    try {

        const data = await fetchAPI("/eda");

        renderTargetChart(data.target);

        renderAgeChart(data.age);

        renderSexChart(data.sex_target);

        renderCPChart(data.cp_target);

        renderCorrelation(data.correlation);

    } catch (error) {

        console.error("EDA API Error:", error);

    }
}


/* ================= TARGET CHART ================= */

function renderTargetChart(target) {

    const container = document.getElementById("targetChart");

    const entries = Object.entries(target);

    const max = Math.max(...entries.map(item => item[1]));

    container.innerHTML = `
        <div class="bar-chart">
            ${entries.map(([label, value]) => {

                const height = (value / max) * 85;

                const name =
                    label === "1"
                        ? "Disease Present"
                        : "No Disease";

                return `
                    <div class="bar-item">

                        <div class="bar-value">
                            ${value}
                        </div>

                        <div class="bar"
                             style="height:${height}%">
                        </div>

                        <div class="bar-label">
                            ${name}
                        </div>

                    </div>
                `;

            }).join("")}
        </div>
    `;
}


/* ================= AGE CHART ================= */

function renderAgeChart(ageData) {

    const container = document.getElementById("ageChart");

    const min = Math.min(...ageData);
    const max = Math.max(...ageData);

    const bins = 12;

    const step = (max - min + 1) / bins;

    const counts = Array(bins).fill(0);

    ageData.forEach(age => {

        let index = Math.floor((age - min) / step);

        if (index >= bins) {
            index = bins - 1;
        }

        counts[index]++;

    });

    const maxCount = Math.max(...counts);

    container.innerHTML = `
        <div class="histogram">

            ${counts.map(count => {

                const height = (count / maxCount) * 90;

                return `
                    <div
                        class="hist-bar"
                        style="height:${height}%"
                        title="${count} records">
                    </div>
                `;

            }).join("")}

        </div>
    `;
}


/* ================= SEX CHART ================= */

function renderSexChart(data) {

    const container = document.getElementById("sexChart");

    const values = [];

    Object.entries(data).forEach(([sex, targets]) => {

        values.push({
            label: sex === "1" ? "Male" : "Female",
            zero: targets["0"] || 0,
            one: targets["1"] || 0
        });

    });

    const max = Math.max(
        ...values.flatMap(v => [v.zero, v.one])
    );

    container.innerHTML = `
        <div class="group-chart">

            ${values.map(item => {

                const h0 = (item.zero / max) * 90;
                const h1 = (item.one / max) * 90;

                return `
                    <div style="
                        display:flex;
                        flex-direction:column;
                        align-items:center;
                        height:100%;
                        justify-content:flex-end;
                    ">

                        <div class="group">

                            <div
                                class="group-bar a"
                                style="height:${h0}%"
                                title="No disease: ${item.zero}">
                            </div>

                            <div
                                class="group-bar b"
                                style="height:${h1}%"
                                title="Disease: ${item.one}">
                            </div>

                        </div>

                        <span
                            style="
                                margin-top:8px;
                                font-size:9px;
                                color:#64748b;
                            ">
                            ${item.label}
                        </span>

                    </div>
                `;

            }).join("")}

        </div>

        <div style="
            display:flex;
            justify-content:center;
            gap:20px;
            margin-top:8px;
            font-size:9px;
            color:#64748b;
        ">
            <span>■ No Disease</span>
            <span style="color:#93c5fd">■ Disease</span>
        </div>
    `;
}


/* ================= CP CHART ================= */

function renderCPChart(data) {

    const container = document.getElementById("cpChart");

    const entries = Object.entries(data);

    const max = Math.max(
        ...entries.flatMap(
            ([, value]) => [
                value["0"] || 0,
                value["1"] || 0
            ]
        )
    );

    container.innerHTML = `
        <div class="group-chart">

            ${entries.map(([cp, values]) => {

                const zero = values["0"] || 0;
                const one = values["1"] || 0;

                const h0 = (zero / max) * 90;
                const h1 = (one / max) * 90;

                return `
                    <div style="
                        display:flex;
                        flex-direction:column;
                        align-items:center;
                        height:100%;
                        justify-content:flex-end;
                    ">

                        <div class="group">

                            <div
                                class="group-bar a"
                                style="height:${h0}%"
                                title="No disease: ${zero}">
                            </div>

                            <div
                                class="group-bar b"
                                style="height:${h1}%"
                                title="Disease: ${one}">
                            </div>

                        </div>

                        <span style="
                            margin-top:8px;
                            font-size:9px;
                            color:#64748b;
                        ">
                            CP ${cp}
                        </span>

                    </div>
                `;

            }).join("")}

        </div>

        <div style="
            display:flex;
            justify-content:center;
            gap:20px;
            margin-top:8px;
            font-size:9px;
            color:#64748b;
        ">
            <span>■ No Disease</span>
            <span style="color:#93c5fd">■ Disease</span>
        </div>
    `;
}


/* ================= CORRELATION ================= */

function renderCorrelation(data) {

    const container =
        document.getElementById("correlationChart");

    const columns = data.columns;
    const values = data.values;

    let html = `
        <div
            class="heatmap"
            style="
                grid-template-columns:
                85px repeat(${columns.length}, 1fr);
            "
        >
            <div></div>
    `;

    columns.forEach(column => {

        html += `
            <div class="heat-cell heat-label">
                ${column}
            </div>
        `;

    });

    values.forEach((row, i) => {

        html += `
            <div class="heat-cell heat-label">
                ${columns[i]}
            </div>
        `;

        row.forEach(value => {

            const v = Number(value);

            const intensity = Math.round(
                Math.abs(v) * 210
            );

            let background;

            if (v >= 0) {
                background =
                    `rgb(${235 - intensity / 3},
                         ${245 - intensity / 5},
                         255)`;
            } else {
                background =
                    `rgb(255,
                         ${245 - intensity / 5},
                         ${245 - intensity / 3})`;
            }

            html += `
                <div
                    class="heat-cell"
                    style="background:${background}"
                    title="${v.toFixed(2)}"
                >
                    ${v.toFixed(2)}
                </div>
            `;

        });

    });

    html += `</div>`;

    container.innerHTML = html;
}


/* ================= CONFUSION MATRIX ================= */

function renderConfusionMatrix(matrix) {

    const container =
        document.getElementById("confusionMatrix");

    if (!matrix || matrix.length !== 2) {
        container.innerHTML = "No confusion matrix available.";
        return;
    }

    container.innerHTML = `
        <div>

            <div style="
                text-align:center;
                margin-bottom:10px;
                color:#64748b;
                font-size:9px;
            ">
                Actual vs Predicted
            </div>

            <div class="confusion-grid">

                <div class="confusion-cell">
                    <strong>${matrix[0][0]}</strong>
                    <span>True Negative</span>
                </div>

                <div class="confusion-cell">
                    <strong>${matrix[0][1]}</strong>
                    <span>False Positive</span>
                </div>

                <div class="confusion-cell">
                    <strong>${matrix[1][0]}</strong>
                    <span>False Negative</span>
                </div>

                <div class="confusion-cell">
                    <strong>${matrix[1][1]}</strong>
                    <span>True Positive</span>
                </div>

            </div>

        </div>
    `;
}


/* ================= FEATURE IMPORTANCE ================= */

function renderFeatureImportance(data) {

    const container =
        document.getElementById("featureImportance");

    if (!data) {
        container.innerHTML = "No feature importance available.";
        return;
    }

    const entries = Object.entries(data);

    const max = Math.max(
        ...entries.map(item => item[1])
    );

    container.innerHTML = entries.map(
        ([feature, value]) => {

            const width =
                (value / max) * 100;

            return `
                <div class="importance-row">

                    <label>${feature}</label>

                    <div class="importance-track">

                        <div
                            class="importance-bar"
                            style="width:${width}%">
                        </div>

                    </div>

                    <span class="importance-value">
                        ${(value * 100).toFixed(1)}%
                    </span>

                </div>
            `;

        }
    ).join("");
}


/* ================= PREDICTION ================= */

document
    .getElementById("predictionForm")
    .addEventListener("submit", async function(event) {

        event.preventDefault();

        const button =
            document.querySelector(".predict-btn");

        button.disabled = true;

        button.querySelector("span").textContent =
            "Running Model...";


        const data = {

            age: Number(document.getElementById("age").value),

            sex: Number(document.getElementById("sex").value),

            cp: Number(document.getElementById("cp").value),

            trestbps:
                Number(document.getElementById("trestbps").value),

            chol:
                Number(document.getElementById("chol").value),

            fbs:
                Number(document.getElementById("fbs").value),

            restecg:
                Number(document.getElementById("restecg").value),

            thalach:
                Number(document.getElementById("thalach").value),

            exang:
                Number(document.getElementById("exang").value),

            oldpeak:
                Number(document.getElementById("oldpeak").value),

            slope:
                Number(document.getElementById("slope").value),

            ca:
                Number(document.getElementById("ca").value),

            thal:
                Number(document.getElementById("thal").value)
        };


        try {

            const response = await fetch(
                `${API_BASE}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(data)
                }
            );


            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed");
            }


            document
                .getElementById("resultPlaceholder")
                .style.display = "none";

            const resultBox =
                document.getElementById("result");

            resultBox.style.display = "block";


            document.getElementById("resultTitle").textContent =
                result.result;


            document.getElementById("resultText").textContent =
                `Model prediction class: ${result.prediction}`;


            document.getElementById("probabilityValue").textContent =
                `${result.probability}%`;


            setTimeout(() => {

                document.getElementById("probabilityBar")
                    .style.width =
                    `${result.probability}%`;

            }, 100);


        } catch (error) {

            alert(
                "Prediction failed.\n\n" +
                error.message
            );

        } finally {

            button.disabled = false;

            button.querySelector("span").textContent =
                "Run ML Prediction";
        }

    });


/* ================= INITIALIZE ================= */

async function initializeApp() {

    try {

        await fetchAPI("/health");

        console.log("CardioAI API connected.");

    } catch (error) {

        console.error(
            "Backend connection failed.",
            error
        );

    }

    await Promise.all([
        loadModel(),
        loadEDA()
    ]);
}


initializeApp();