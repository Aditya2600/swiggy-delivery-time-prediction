import { useState } from "react";

const weatherOptions = [
  { label: "Sunny", value: "conditions Sunny" },
  { label: "Cloudy", value: "conditions Cloudy" },
  { label: "Fog", value: "conditions Fog" },
  { label: "Stormy", value: "conditions Stormy" },
  { label: "Windy", value: "conditions Windy" },
  { label: "Sandstorms", value: "conditions Sandstorms" },
  { label: "Unknown", value: "conditions NaN" },
];

const trafficOptions = [
  { label: "Low", value: "Low" },
  { label: "Medium", value: "Medium" },
  { label: "High", value: "High" },
  { label: "Jam", value: "Jam" },
  { label: "Unknown", value: "NaN" },
];

const orderTypeOptions = [
  { label: "Meal", value: "Meal" },
  { label: "Snack", value: "Snack" },
  { label: "Drinks", value: "Drinks" },
  { label: "Buffet", value: "Buffet" },
];

const vehicleTypeOptions = [
  { label: "Motorcycle", value: "motorcycle" },
  { label: "Scooter", value: "scooter" },
  { label: "Electric Scooter", value: "electric_scooter" },
  { label: "Bicycle", value: "bicycle" },
];

const festivalOptions = [
  { label: "No", value: "No" },
  { label: "Yes", value: "Yes" },
  { label: "Unknown", value: "NaN" },
];

const cityOptions = [
  { label: "Urban", value: "Urban" },
  { label: "Semi-Urban", value: "Semi-Urban" },
  { label: "Metropolitian", value: "Metropolitian" },
  { label: "Unknown", value: "NaN" },
];

const multipleDeliveryOptions = [
  { label: "0", value: "0" },
  { label: "1", value: "1" },
  { label: "2", value: "2" },
  { label: "3", value: "3" },
  { label: "Unknown", value: "NaN" },
];

const vehicleConditionOptions = [
  { label: "0", value: "0" },
  { label: "1", value: "1" },
  { label: "2", value: "2" },
  { label: "3", value: "3" },
];

const sampleOrder = {
  ID: "0x4607",
  Delivery_person_ID: "INDORES13DEL02",
  Delivery_person_Age: "37",
  Delivery_person_Ratings: "4.9",
  Restaurant_latitude: "22.745049",
  Restaurant_longitude: "75.892471",
  Delivery_location_latitude: "22.765049",
  Delivery_location_longitude: "75.912471",
  Order_Date: "2022-03-19",
  Time_Orderd: "11:30",
  Time_Order_picked: "11:45",
  Weatherconditions: "conditions Sunny",
  Road_traffic_density: "High",
  Vehicle_condition: "2",
  Type_of_order: "Snack",
  Type_of_vehicle: "motorcycle",
  multiple_deliveries: "0",
  Festival: "No",
  City: "Urban",
};

const fieldGroups = [
  {
    title: "Order Details",
    subtitle: "Identity, time, and order load.",
    fields: [
      { name: "ID", label: "Order ID", type: "text" },
      { name: "Order_Date", label: "Order Date", type: "date" },
      { name: "Time_Orderd", label: "Order Time", type: "time" },
      { name: "Time_Order_picked", label: "Picked Time", type: "time" },
      {
        name: "multiple_deliveries",
        label: "Multiple Deliveries",
        type: "select",
        options: multipleDeliveryOptions,
      },
      {
        name: "Festival",
        label: "Festival",
        type: "select",
        options: festivalOptions,
      },
    ],
  },
  {
    title: "Rider Profile",
    subtitle: "Ratings and vehicle condition.",
    fields: [
      { name: "Delivery_person_ID", label: "Rider ID", type: "text" },
      { name: "Delivery_person_Age", label: "Rider Age", type: "number", step: "1" },
      {
        name: "Delivery_person_Ratings",
        label: "Rider Rating",
        type: "number",
        step: "0.1",
      },
      {
        name: "Vehicle_condition",
        label: "Vehicle Condition",
        type: "select",
        options: vehicleConditionOptions,
      },
    ],
  },
  {
    title: "Locations",
    subtitle: "Pick-up and drop-off coordinates.",
    fields: [
      {
        name: "Restaurant_latitude",
        label: "Restaurant Latitude",
        type: "number",
        step: "0.000001",
      },
      {
        name: "Restaurant_longitude",
        label: "Restaurant Longitude",
        type: "number",
        step: "0.000001",
      },
      {
        name: "Delivery_location_latitude",
        label: "Delivery Latitude",
        type: "number",
        step: "0.000001",
      },
      {
        name: "Delivery_location_longitude",
        label: "Delivery Longitude",
        type: "number",
        step: "0.000001",
      },
    ],
  },
  {
    title: "Context",
    subtitle: "Traffic, weather, and order type.",
    fields: [
      {
        name: "Weatherconditions",
        label: "Weather",
        type: "select",
        options: weatherOptions,
      },
      {
        name: "Road_traffic_density",
        label: "Traffic",
        type: "select",
        options: trafficOptions,
      },
      {
        name: "Type_of_order",
        label: "Order Type",
        type: "select",
        options: orderTypeOptions,
      },
      {
        name: "Type_of_vehicle",
        label: "Vehicle Type",
        type: "select",
        options: vehicleTypeOptions,
      },
      {
        name: "City",
        label: "City",
        type: "select",
        options: cityOptions,
      },
    ],
  },
];

const normalizeDate = (value) => {
  if (!value) {
    return "";
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split("-");
    return `${day}-${month}-${year}`;
  }
  return value;
};

const normalizeTime = (value) => {
  if (!value) {
    return "";
  }
  if (/^\d{2}:\d{2}$/.test(value)) {
    return `${value}:00`;
  }
  return value;
};

const parsePrediction = (payload) => {
  if (typeof payload === "number") {
    return payload;
  }
  if (typeof payload === "string") {
    return Number(payload);
  }
  if (payload && typeof payload === "object") {
    if ("prediction" in payload) {
      return Number(payload.prediction);
    }
  }
  return Number.NaN;
};

const buildPayload = (form) => ({
  ...form,
  Order_Date: normalizeDate(form.Order_Date),
  Time_Orderd: normalizeTime(form.Time_Orderd),
  Time_Order_picked: normalizeTime(form.Time_Order_picked),
  Restaurant_latitude: Number(form.Restaurant_latitude),
  Restaurant_longitude: Number(form.Restaurant_longitude),
  Delivery_location_latitude: Number(form.Delivery_location_latitude),
  Delivery_location_longitude: Number(form.Delivery_location_longitude),
  Vehicle_condition: Number(form.Vehicle_condition),
});

const formatValue = (value) => {
  if (!value) {
    return "--";
  }
  return value
    .replace(/^conditions\s*/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
};

export default function App() {
  const [form, setForm] = useState(sampleOrder);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
    /\/$/,
    ""
  );

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSample = () => {
    setForm(sampleOrder);
    setResult(null);
    setStatus("idle");
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus("loading");
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(buildPayload(form)),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Request failed (${response.status}).`);
      }

      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

      const prediction = parsePrediction(payload);
      if (Number.isNaN(prediction)) {
        throw new Error("Unexpected response from the prediction API.");
      }

      setResult(prediction);
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(err?.message || "Prediction failed. Check the API and inputs.");
    }
  };

  const etaValue = result !== null ? result.toFixed(2) : "--";
  const meterValue = result !== null ? Math.min(100, Math.max(8, (result / 80) * 100)) : 12;
  const statusLabel =
    status === "loading"
      ? "Calculating ETA..."
      : status === "success"
      ? "Prediction ready"
      : status === "error"
      ? "Needs attention"
      : "Waiting for input";

  const statusClass =
    status === "loading" ? "status-dot status-dot--loading" :
    status === "success" ? "status-dot status-dot--success" :
    status === "error" ? "status-dot status-dot--error" :
    "status-dot";

  const highlights = [
    { label: "Traffic", value: formatValue(form.Road_traffic_density) },
    { label: "Order", value: formatValue(form.Type_of_order) },
    { label: "Vehicle", value: formatValue(form.Type_of_vehicle) },
    { label: "City", value: formatValue(form.City) },
  ];

  return (
    <div className="app">
      <div className="ambient">
        <span className="orb orb--one" />
        <span className="orb orb--two" />
        <span className="orb orb--three" />
      </div>

      <header className="topbar">
        <div className="logo">
          <span className="logo-mark">S</span>
          <div>
            <div className="logo-title">Swiggy ETA Studio</div>
            <div className="logo-subtitle">Delivery prediction cockpit</div>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="pill">StackingRegressor</span>
          <span className="pill pill--alt">MLflow + FastAPI</span>
        </div>
      </header>

      <main className="layout">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">Food delivery forecasting</p>
            <h1>Turn raw order details into a cinematic ETA in seconds.</h1>
            <p className="lede">
              Feed the model with live order metadata, see the ETA surface instantly, and
              calibrate delivery strategy with confidence. Designed for fast ops teams.
            </p>
            <div className="hero-actions">
              <button className="ghost-button" type="button" onClick={handleSample}>
                Load sample order
              </button>
              <div className="api-pill">API: {apiBaseUrl}</div>
            </div>
            <div className="hero-chips">
              <span className="chip">Realtime geospatial</span>
              <span className="chip">Traffic-aware</span>
              <span className="chip">Weather-smart</span>
              <span className="chip">Batch-ready</span>
            </div>
          </div>
          <div className="hero-panel">
            <div className="stat-card">
              <div className="stat-title">Model status</div>
              <div className="stat-value">Production</div>
              <div className="stat-meta">Registry tracked via DagsHub</div>
            </div>
            <div className="stat-card stat-card--accent">
              <div className="stat-title">Avg pickup time</div>
              <div className="stat-value">~14 min</div>
              <div className="stat-meta">Derived from engineered signals</div>
            </div>
            <div className="stat-card">
              <div className="stat-title">Ops insight</div>
              <div className="stat-value">Route + rider blend</div>
              <div className="stat-meta">Distance + rating + traffic</div>
            </div>
          </div>
        </section>

        <section className="content">
          <form className="form-card" onSubmit={handleSubmit}>
            <div className="form-header">
              <div>
                <h2>Order input panel</h2>
                <p>Complete the fields to predict delivery time.</p>
              </div>
              <button className="secondary-button" type="button" onClick={handleSample}>
                Reset sample
              </button>
            </div>

            {fieldGroups.map((group, groupIndex) => (
              <fieldset className="field-group" key={group.title}>
                <legend>
                  <span>{group.title}</span>
                  <small>{group.subtitle}</small>
                </legend>
                <div className="field-grid">
                  {group.fields.map((field, fieldIndex) => (
                    <div
                      className="field"
                      key={field.name}
                      style={{ "--delay": `${(groupIndex * 6 + fieldIndex) * 0.04}s` }}
                    >
                      <label htmlFor={field.name}>{field.label}</label>
                      {field.type === "select" ? (
                        <select
                          id={field.name}
                          name={field.name}
                          value={form[field.name]}
                          onChange={handleChange}
                          required
                        >
                          {field.options.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id={field.name}
                          name={field.name}
                          type={field.type}
                          value={form[field.name]}
                          onChange={handleChange}
                          step={field.step}
                          required
                        />
                      )}
                    </div>
                  ))}
                </div>
              </fieldset>
            ))}

            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={status === "loading"}>
                {status === "loading" ? "Predicting..." : "Predict ETA"}
              </button>
              {error ? <span className="form-error">{error}</span> : null}
            </div>
          </form>

          <aside className="results-card">
            <div className="results-header">
              <div className={statusClass} />
              <div>
                <div className="results-title">Prediction signal</div>
                <div className="results-subtitle">{statusLabel}</div>
              </div>
            </div>

            <div className="eta-display">
              <div className="eta-value">{etaValue}</div>
              <div className="eta-unit">minutes</div>
            </div>

            <div className="eta-meter">
              <div className="eta-meter-fill" style={{ width: `${meterValue}%` }} />
            </div>

            <div className="results-grid">
              {highlights.map((item) => (
                <div className="mini-card" key={item.label}>
                  <div className="mini-label">{item.label}</div>
                  <div className="mini-value">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="results-foot">
              <div className="results-note">Live endpoint at /predict</div>
              <div className="results-note">Predictions update on submit</div>
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}
