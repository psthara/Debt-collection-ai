import { useMemo, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

// ============================================================
// SAMPLE DATA
// ============================================================

function generateData() {
  const channels = ["PHONE", "SMS", "WHATSAPP", "EMAIL"];
  const dpdValues = [0, 5, 15, 30, 45, 60, 90, 120, 180];

  const data = [];

  for (let i = 0; i < 500; i++) {
    const age = randomInt(21, 69);
    const income = randomInt(20000, 150000);
    const loanAmount = randomInt(50000, 1500000);
    const outstanding = randomInt(10000, 800000);
    const emi = randomInt(2000, 50000);
    const creditScore = randomInt(450, 849);

    const dpd =
      dpdValues[randomInt(0, dpdValues.length - 1)];

    const missedPayments = randomInt(0, 6);
    const previousPTP = randomInt(0, 6);
    const previousPTPKept = randomInt(
      0,
      Math.min(5, previousPTP)
    );

    const collectionAttempts = randomInt(0, 11);
    const successfulContacts = randomInt(0, 7);
    const recentPayment = randomInt(0, 29999);
    const daysSincePayment = randomInt(0, 179);

    const preferredChannel =
      channels[randomInt(0, channels.length - 1)];

    // --------------------------------------------------------
    // Simulated AI predictions
    // --------------------------------------------------------

    const stress =
      0.40 * Math.min(dpd / 180, 1) +
      0.20 * Math.min(missedPayments / 6, 1) +
      0.20 * Math.min(daysSincePayment / 180, 1) +
      0.20 *
        (1 -
          previousPTPKept /
            Math.max(previousPTP, 1));

    const paymentProbability = clamp(
      0.95 - stress + randomNormal(0, 0.05),
      0.02,
      0.98
    );

    const ptpProbability = clamp(
      0.90 -
        stress * 0.9 +
        randomNormal(0, 0.05),
      0.02,
      0.98
    );

    const recoveryProbability = clamp(
      0.85 -
        stress * 0.7 +
        randomNormal(0, 0.05),
      0.02,
      0.98
    );

    // --------------------------------------------------------
    // Priority
    // --------------------------------------------------------

    let priorityScore =
      (1 - paymentProbability) * 40 +
      (1 - ptpProbability) * 25 +
      Math.min(dpd / 180, 1) * 25 +
      Math.min(
        outstanding / Math.max(income * 12, 1),
        1
      ) *
        10;

    priorityScore = clamp(
      priorityScore,
      0,
      100
    );

    let priority = "LOW";

    if (priorityScore >= 70) {
      priority = "HIGH";
    } else if (priorityScore >= 40) {
      priority = "MEDIUM";
    }

    // --------------------------------------------------------
    // Next Best Action
    // --------------------------------------------------------

    let nextBestAction;

    if (dpd >= 90) {
      nextBestAction = "Human Review";
    } else if (ptpProbability >= 0.65) {
      nextBestAction = "Request PTP";
    } else if (paymentProbability >= 0.60) {
      nextBestAction = "Repayment Discussion";
    } else {
      nextBestAction = "Collection Call";
    }

    data.push({
      customer_id: `C${1000 + i}`,
      age,
      income,
      loan_amount: loanAmount,
      outstanding_amount: outstanding,
      emi_amount: emi,
      credit_score: creditScore,
      dpd,
      missed_payment_count: missedPayments,
      previous_ptp_count: previousPTP,
      previous_ptp_kept_count: previousPTPKept,
      collection_attempts: collectionAttempts,
      successful_contacts: successfulContacts,
      recent_payment_amount: recentPayment,
      days_since_last_payment: daysSincePayment,
      preferred_channel: preferredChannel,

      payment_probability: paymentProbability,
      ptp_probability: ptpProbability,
      recovery_probability: recoveryProbability,

      priority_score: priorityScore,
      priority,
      next_best_action: nextBestAction,
    });
  }

  return data;
}

// ============================================================
// HELPERS
// ============================================================

function randomInt(min, max) {
  return Math.floor(
    Math.random() * (max - min + 1)
  ) + min;
}

function randomNormal(mean = 0, std = 1) {
  let u = 0;
  let v = 0;

  while (u === 0) {
    u = Math.random();
  }

  while (v === 0) {
    v = Math.random();
  }

  return (
    mean +
    std *
      Math.sqrt(-2 * Math.log(u)) *
      Math.cos(2 * Math.PI * v)
  );
}

function clamp(value, min, max) {
  return Math.min(
    Math.max(value, min),
    max
  );
}

function formatCurrency(value) {
  return `₹${Number(value).toLocaleString("en-IN")}`;
}

// ============================================================
// APP
// ============================================================

function App() {
  const [data] = useState(() =>
    generateData()
  );

  const [priorityFilter, setPriorityFilter] =
    useState([
      "HIGH",
      "MEDIUM",
      "LOW",
    ]);

  const [dpdFilter, setDpdFilter] =
    useState(0);

  const [channelFilter, setChannelFilter] =
    useState([
      "PHONE",
      "SMS",
      "WHATSAPP",
      "EMAIL",
    ]);

  const [selectedCustomerId, setSelectedCustomerId] =
    useState("");

  const [question, setQuestion] =
    useState("");

  const [ragAnswer, setRagAnswer] =
    useState("");

  const [ragSources, setRagSources] =
    useState([]);

  const [tone, setTone] =
    useState("professional");

  const [generatedMessage, setGeneratedMessage] =
    useState("");

  // ==========================================================
  // FILTER DATA
  // ==========================================================

  const filteredData = useMemo(() => {
    return data.filter(
      (customer) =>
        priorityFilter.includes(
          customer.priority
        ) &&
        customer.dpd >= dpdFilter &&
        channelFilter.includes(
          customer.preferred_channel
        )
    );
  }, [
    data,
    priorityFilter,
    dpdFilter,
    channelFilter,
  ]);

  // ==========================================================
  // KPI
  // ==========================================================

  const totalCustomers =
    filteredData.length;

  const totalOutstanding =
    filteredData.reduce(
      (sum, customer) =>
        sum +
        customer.outstanding_amount,
      0
    );

  const highRisk =
    filteredData.filter(
      (c) => c.priority === "HIGH"
    ).length;

  const avgPaymentProbability =
    average(
      filteredData.map(
        (c) => c.payment_probability
      )
    );

  const avgRecoveryProbability =
    average(
      filteredData.map(
        (c) => c.recovery_probability
      )
    );

  // ==========================================================
  // CHART DATA
  // ==========================================================

  const riskData = [
    {
      name: "HIGH",
      count: filteredData.filter(
        (c) => c.priority === "HIGH"
      ).length,
    },
    {
      name: "MEDIUM",
      count: filteredData.filter(
        (c) => c.priority === "MEDIUM"
      ).length,
    },
    {
      name: "LOW",
      count: filteredData.filter(
        (c) => c.priority === "LOW"
      ).length,
    },
  ];

  const actionNames = [
    "Human Review",
    "Request PTP",
    "Repayment Discussion",
    "Collection Call",
  ];

  const actionData =
    actionNames.map((action) => ({
      name: action,
      count: filteredData.filter(
        (c) =>
          c.next_best_action === action
      ).length,
    }));

  // ==========================================================
  // CUSTOMER
  // ==========================================================

  const selectedCustomer =
    filteredData.find(
      (c) =>
        c.customer_id ===
        selectedCustomerId
    ) || filteredData[0];

  // ==========================================================
  // RAG
  // ==========================================================

  async function askRAG() {
    if (!question.trim()) return;

    try {
      const response =
        await axios.post(
          `${API_URL}/rag/query`,
          {
            question,
          },
          {
            timeout: 60000,
          }
        );

      setRagAnswer(
        response.data.answer ||
          "No answer"
      );

      setRagSources(
        response.data.sources || []
      );
    } catch (error) {
      setRagAnswer(
        "Could not connect to FastAPI."
      );

      setRagSources([]);
    }
  }

  // ==========================================================
  // GENAI MESSAGE
  // ==========================================================

  async function generateMessage() {
    if (!selectedCustomer) return;

    const customerPayload = {
      customer_id:
        selectedCustomer.customer_id,

      age:
        selectedCustomer.age,

      income:
        selectedCustomer.income,

      loan_amount:
        selectedCustomer.loan_amount,

      outstanding_amount:
        selectedCustomer.outstanding_amount,

      emi_amount:
        selectedCustomer.emi_amount,

      credit_score:
        selectedCustomer.credit_score,

      dpd:
        selectedCustomer.dpd,

      missed_payment_count:
        selectedCustomer.missed_payment_count,

      previous_ptp_count:
        selectedCustomer.previous_ptp_count,

      previous_ptp_kept_count:
        selectedCustomer.previous_ptp_kept_count,

      collection_attempts:
        selectedCustomer.collection_attempts,

      successful_contacts:
        selectedCustomer.successful_contacts,

      recent_payment_amount:
        selectedCustomer.recent_payment_amount,

      days_since_last_payment:
        selectedCustomer.days_since_last_payment,

      preferred_channel:
        selectedCustomer.preferred_channel,
    };

    try {
      const response =
        await axios.post(
          `${API_URL}/generate-message`,
          {
            customer:
              customerPayload,
            tone,
          },
          {
            timeout: 60000,
          }
        );

      setGeneratedMessage(
        response.data.message || ""
      );
    } catch (error) {
      setGeneratedMessage(
        "Could not connect to FastAPI."
      );
    }
  }

  return (
    <div className="app">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <header>
        <h1>
          💳 AI Debt Collection Intelligence
        </h1>

        <p>
          ML + Next Best Action + RAG + GenAI
        </p>
      </header>

      {/* =====================================================
          SIDEBAR / FILTERS
      ====================================================== */}

      <aside className="sidebar">

        <h2>🔎 Filters</h2>

        <label>Priority</label>

        {["HIGH", "MEDIUM", "LOW"].map(
          (priority) => (
            <label
              key={priority}
              className="checkbox"
            >
              <input
                type="checkbox"
                checked={priorityFilter.includes(
                  priority
                )}
                onChange={() => {
                  setPriorityFilter((prev) =>
                    prev.includes(priority)
                      ? prev.filter(
                          (x) =>
                            x !== priority
                        )
                      : [
                          ...prev,
                          priority,
                        ]
                  );
                }}
              />

              {priority}
            </label>
          )
        )}

        <label>
          Minimum DPD: {dpdFilter}
        </label>

        <input
          type="range"
          min="0"
          max="180"
          value={dpdFilter}
          onChange={(e) =>
            setDpdFilter(
              Number(e.target.value)
            )
          }
        />

        <label>Preferred Channel</label>

        {[
          "PHONE",
          "SMS",
          "WHATSAPP",
          "EMAIL",
        ].map((channel) => (
          <label
            key={channel}
            className="checkbox"
          >
            <input
              type="checkbox"
              checked={channelFilter.includes(
                channel
              )}
              onChange={() => {
                setChannelFilter((prev) =>
                  prev.includes(channel)
                    ? prev.filter(
                        (x) => x !== channel
                      )
                    : [...prev, channel]
                );
              }}
            />

            {channel}
          </label>
        ))}
      </aside>

      {/* =====================================================
          MAIN
      ====================================================== */}

      <main>

        {/* KPI */}

        <section className="kpi-grid">

          <KPI
            title="Customers"
            value={totalCustomers.toLocaleString()}
          />

          <KPI
            title="Outstanding"
            value={`₹${(
              totalOutstanding / 10000000
            ).toFixed(2)} Cr`}
          />

          <KPI
            title="High Risk"
            value={highRisk.toLocaleString()}
          />

          <KPI
            title="Avg Payment Probability"
            value={`${(
              avgPaymentProbability * 100
            ).toFixed(1)}%`}
          />

          <KPI
            title="Avg Recovery Probability"
            value={`${(
              avgRecoveryProbability * 100
            ).toFixed(1)}%`}
          />

        </section>

        {/* =================================================
            CHARTS
        ================================================== */}

        <section>

          <h2>
            📊 AI Risk Distribution
          </h2>

          <div className="charts">

            <div className="card">

              <h3>Risk Distribution</h3>

              <ResponsiveContainer
                width="100%"
                height={300}
              >
                <BarChart
                  data={riskData}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                  />

                  <XAxis dataKey="name" />

                  <YAxis />

                  <Tooltip />

                  <Bar
                    dataKey="count"
                    fill="#2563eb"
                  />
                </BarChart>
              </ResponsiveContainer>

            </div>

            <div className="card">

              <h3>Next Best Action</h3>

              <ResponsiveContainer
                width="100%"
                height={300}
              >
                <BarChart
                  data={actionData}
                  layout="vertical"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                  />

                  <XAxis type="number" />

                  <YAxis
                    dataKey="name"
                    type="category"
                    width={150}
                  />

                  <Tooltip />

                  <Bar
                    dataKey="count"
                    fill="#16a34a"
                  />
                </BarChart>
              </ResponsiveContainer>

            </div>

          </div>

        </section>

        {/* =================================================
            PREDICTIONS
        ================================================== */}

        <section>

          <h2>
            🤖 AI Prediction Summary
          </h2>

          <div className="prediction-grid">

            <KPI
              title="Payment Probability"
              value={`${(
                avgPaymentProbability *
                100
              ).toFixed(1)}%`}
            />

            <KPI
              title="PTP Probability"
              value={`${(
                average(
                  filteredData.map(
                    (c) =>
                      c.ptp_probability
                  )
                ) * 100
              ).toFixed(1)}%`}
            />

            <KPI
              title="Recovery Probability"
              value={`${(
                avgRecoveryProbability *
                100
              ).toFixed(1)}%`}
            />

          </div>

        </section>

        {/* =================================================
            PRIORITY TABLE
        ================================================== */}

        <section>

          <h2>
            🚨 AI Collection Priority Queue
          </h2>

          <div className="table-container">

            <table>

              <thead>

                <tr>

                  <th>Customer</th>
                  <th>DPD</th>
                  <th>Outstanding</th>
                  <th>Payment</th>
                  <th>PTP</th>
                  <th>Recovery</th>
                  <th>Priority Score</th>
                  <th>Priority</th>
                  <th>Next Action</th>
                  <th>Channel</th>

                </tr>

              </thead>

              <tbody>

                {filteredData
                  .sort(
                    (a, b) =>
                      b.priority_score -
                      a.priority_score
                  )
                  .slice(0, 50)
                  .map((customer) => (

                    <tr
                      key={
                        customer.customer_id
                      }
                    >

                      <td>
                        {
                          customer.customer_id
                        }
                      </td>

                      <td>
                        {customer.dpd}
                      </td>

                      <td>
                        {formatCurrency(
                          customer.outstanding_amount
                        )}
                      </td>

                      <td>
                        {(
                          customer.payment_probability *
                          100
                        ).toFixed(1)}
                        %
                      </td>

                      <td>
                        {(
                          customer.ptp_probability *
                          100
                        ).toFixed(1)}
                        %
                      </td>

                      <td>
                        {(
                          customer.recovery_probability *
                          100
                        ).toFixed(1)}
                        %
                      </td>

                      <td>
                        {customer.priority_score.toFixed(
                          1
                        )}
                      </td>

                      <td>
                        <span
                          className={`priority ${customer.priority.toLowerCase()}`}
                        >
                          {
                            customer.priority
                          }
                        </span>
                      </td>

                      <td>
                        {
                          customer.next_best_action
                        }
                      </td>

                      <td>
                        {
                          customer.preferred_channel
                        }
                      </td>

                    </tr>

                  ))}

              </tbody>

            </table>

          </div>

        </section>

        {/* =================================================
            CUSTOMER 360
        ================================================== */}

        <section>

          <h2>
            👤 Customer 360
          </h2>

          <select
            value={
              selectedCustomer
                ?.customer_id || ""
            }
            onChange={(e) =>
              setSelectedCustomerId(
                e.target.value
              )
            }
          >

            {filteredData.map(
              (customer) => (
                <option
                  key={
                    customer.customer_id
                  }
                  value={
                    customer.customer_id
                  }
                >
                  {
                    customer.customer_id
                  }
                </option>
              )
            )}

          </select>

          {selectedCustomer && (

            <div className="customer-grid">

              <div className="card">

                <h3>
                  Customer Information
                </h3>

                <p>
                  <b>Customer ID:</b>{" "}
                  {
                    selectedCustomer.customer_id
                  }
                </p>

                <p>
                  <b>Age:</b>{" "}
                  {selectedCustomer.age}
                </p>

                <p>
                  <b>Income:</b>{" "}
                  {formatCurrency(
                    selectedCustomer.income
                  )}
                </p>

                <p>
                  <b>Credit Score:</b>{" "}
                  {
                    selectedCustomer.credit_score
                  }
                </p>

              </div>

              <div className="card">

                <h3>
                  Loan Information
                </h3>

                <p>
                  <b>Loan Amount:</b>{" "}
                  {formatCurrency(
                    selectedCustomer.loan_amount
                  )}
                </p>

                <p>
                  <b>Outstanding:</b>{" "}
                  {formatCurrency(
                    selectedCustomer.outstanding_amount
                  )}
                </p>

                <p>
                  <b>EMI:</b>{" "}
                  {formatCurrency(
                    selectedCustomer.emi_amount
                  )}
                </p>

                <p>
                  <b>DPD:</b>{" "}
                  {selectedCustomer.dpd}
                </p>

              </div>

              <div className="card">

                <h3>
                  AI Insights
                </h3>

                <KPI
                  title="Payment Probability"
                  value={`${(
                    selectedCustomer.payment_probability *
                    100
                  ).toFixed(1)}%`}
                />

                <KPI
                  title="PTP Probability"
                  value={`${(
                    selectedCustomer.ptp_probability *
                    100
                  ).toFixed(1)}%`}
                />

                <KPI
                  title="Recovery Probability"
                  value={`${(
                    selectedCustomer.recovery_probability *
                    100
                  ).toFixed(1)}%`}
                />

              </div>

            </div>

          )}

        </section>

        {/* =================================================
            NEXT BEST ACTION
        ================================================== */}

        {selectedCustomer && (

          <section>

            <h2>
              🎯 AI Next Best Action
            </h2>

            <div
              className={`alert ${selectedCustomer.priority.toLowerCase()}`}
            >

              {selectedCustomer.priority ===
                "HIGH" &&
                "🚨 HIGH PRIORITY CUSTOMER"}

              {selectedCustomer.priority ===
                "MEDIUM" &&
                "⚠️ MEDIUM PRIORITY CUSTOMER"}

              {selectedCustomer.priority ===
                "LOW" &&
                "✅ LOW PRIORITY CUSTOMER"}

            </div>

            <div className="kpi-grid">

              <KPI
                title="Priority Score"
                value={selectedCustomer.priority_score.toFixed(
                  1
                )}
              />

              <KPI
                title="Recommended Channel"
                value={
                  selectedCustomer.preferred_channel
                }
              />

              <KPI
                title="Next Best Action"
                value={
                  selectedCustomer.next_best_action
                }
              />

            </div>

            <div className="recommendation">

              <h3>
                AI Recommendation
              </h3>

              <p>
                Customer{" "}
                {
                  selectedCustomer.customer_id
                }{" "}
                should be handled using:
              </p>

              <p>
                <b>Action:</b>{" "}
                {
                  selectedCustomer.next_best_action
                }
              </p>

              <p>
                <b>Channel:</b>{" "}
                {
                  selectedCustomer.preferred_channel
                }
              </p>

              <p>
                <b>Priority:</b>{" "}
                {
                  selectedCustomer.priority
                }
              </p>

              <p>
                <b>Reason:</b> DPD ={" "}
                {selectedCustomer.dpd} days,
                Payment Probability ={" "}
                {(
                  selectedCustomer.payment_probability *
                  100
                ).toFixed(1)}
                %, PTP Probability ={" "}
                {(
                  selectedCustomer.ptp_probability *
                  100
                ).toFixed(1)}
                %.
              </p>

            </div>

          </section>

        )}

        {/* =================================================
            RAG ASSISTANT
        ================================================== */}

        <section>

          <h2>
            📚 AI Collection Policy Assistant
          </h2>

          <textarea
            placeholder="Ask a question about collection policy"
            value={question}
            onChange={(e) =>
              setQuestion(e.target.value)
            }
          />

          <button onClick={askRAG}>
            Ask AI Policy Assistant
          </button>

          {ragAnswer && (

            <div className="recommendation">

              <h3>AI Answer</h3>

              <p>{ragAnswer}</p>

              {ragSources.length > 0 && (

                <>
                  <h3>Sources</h3>

                  {ragSources.map(
                    (source, index) => (
                      <p key={index}>
                        📄{" "}
                        {source.title}
                      </p>
                    )
                  )}
                </>
              )}

            </div>

          )}

        </section>

        {/* =================================================
            GENAI MESSAGE
        ================================================== */}

        <section>

          <h2>
            ✍️ GenAI Collection Message
          </h2>

          <select
            value={tone}
            onChange={(e) =>
              setTone(e.target.value)
            }
          >

            <option value="professional">
              Professional
            </option>

            <option value="empathetic">
              Empathetic
            </option>

            <option value="concise">
              Concise
            </option>

          </select>

          <button
            onClick={generateMessage}
          >
            Generate Collection Message
          </button>

          {generatedMessage && (

            <div className="recommendation">

              <h3>
                AI Generated Message
              </h3>

              <p>
                {generatedMessage}
              </p>

            </div>

          )}

        </section>

        <footer>

          AI Debt Collection Intelligence
          Platform | ML predictions are
          decision-support signals and must
          operate under approved collection
          policies.

        </footer>

      </main>

    </div>
  );
}

// ============================================================
// KPI COMPONENT
// ============================================================

function KPI({ title, value }) {
  return (
    <div className="kpi">

      <span>{title}</span>

      <strong>{value}</strong>

    </div>
  );
}

// ============================================================
// AVERAGE
// ============================================================

function average(values) {
  if (!values.length) return 0;

  return (
    values.reduce(
      (sum, value) =>
        sum + value,
      0
    ) / values.length
  );
}

export default App;