async function checkNews() {
  const inputText = document.getElementById("newsInput").value;
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-type": "applicaton/json",
    },
    body: JSON.stringify({ input_text: inputText }),
  });

  const data = await response.json9();
  document.getElementById("result").innertext = "Result: " + data.result;
}
