// // === boxplot.js ===
// function drawBoxPlot(dataset, parameter) {
//   const values = dataset.map(d => d.value).sort(d3.ascending);

//   const q1 = d3.quantile(values, 0.25);
//   const median = d3.quantile(values, 0.5);
//   const q3 = d3.quantile(values, 0.75);
//   const interQuantileRange = q3 - q1;
//   const min = d3.min(values);
//   const max = d3.max(values);

//   const margin = {top: 10, right: 30, bottom: 30, left: 50},
//         width = 600 - margin.left - margin.right,
//         height = 400 - margin.top - margin.bottom;

//   const svg = d3.select("#boxplot-container")
//     .append("svg")
//     .attr("width", width + margin.left + margin.right)
//     .attr("height", height + margin.top + margin.bottom)
//     .append("g")
//     .attr("transform", `translate(${margin.left},${margin.top})`);

//   const y = d3.scaleLinear()
//     .domain([d3.min(values), d3.max(values)])
//     .range([height, 0]);

//   svg.call(d3.axisLeft(y));

//   const center = width / 2;
//   const boxWidth = 100;

//   // Vertical line
//   svg.append("line")
//     .attr("x1", center)
//     .attr("x2", center)
//     .attr("y1", y(min))
//     .attr("y2", y(max))
//     .attr("stroke", "black");

//   // Box
//   svg.append("rect")
//     .attr("x", center - boxWidth/2)
//     .attr("y", y(q3))
//     .attr("height", y(q1) - y(q3))
//     .attr("width", boxWidth)
//     .attr("stroke", "black")
//     .style("fill", "steelblue")
//     .style("opacity", 0.5);

//   // Median
//   svg.append("line")
//     .attr("x1", center - boxWidth/2)
//     .attr("x2", center + boxWidth/2)
//     .attr("y1", y(median))
//     .attr("y2", y(median))
//     .attr("stroke", "black");

//   // Min & Max
//   [min, max].forEach(val => {
//     svg.append("line")
//       .attr("x1", center - boxWidth/2)
//       .attr("x2", center + boxWidth/2)
//       .attr("y1", y(val))
//       .attr("y2", y(val))
//       .attr("stroke", "black");
//   });

//   // Tooltip
//   const tooltip = d3.select("body").append("div")
//     .attr("class", "tooltip");

//   // Scatter dots
//   svg.selectAll("circle")
//     .data(dataset)
//     .enter()
//     .append("circle")
//       .attr("cx", () => center + (Math.random() - 0.5) * boxWidth) // jitter
//       .attr("cy", d => y(d.value))
//       .attr("r", 3)
//       .style("fill", "steelblue")
//       .style("opacity", 0.6)
//       .on("mouseover", function (event, d) {
//         tooltip.style("display", "block")
//           .html(`<b>${d.date.toISOString().split("T")[0]}</b><br/>${d.value}`);
//       })
//       .on("mousemove", function (event) {
//         tooltip.style("left", (event.pageX + 10) + "px")
//                .style("top", (event.pageY - 20) + "px");
//       })
//       .on("mouseout", function () {
//         tooltip.style("display", "none");
//       });

//   // Add title label
//   svg.append("text")
//     .attr("x", center)
//     .attr("y", -5)
//     .attr("text-anchor", "middle")
//     .style("font-size", "14px")
//     .text(`Boxplot of ${parameter}`);
// }






function drawBoxPlot(dataset, parameter) {
  const values = dataset.map(d => d.value).sort(d3.ascending);

  const q1 = d3.quantile(values, 0.25);
  const median = d3.quantile(values, 0.5);
  const q3 = d3.quantile(values, 0.75);
  const min = d3.min(values);
  const max = d3.max(values);

  const margin = {top: 40, right: 30, bottom: 30, left: 50},
        width = 600 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

  const svg = d3.select("#boxplot-container")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const y = d3.scaleLinear()
    .domain([d3.min(values), d3.max(values)]).nice()
    .range([height, 0]);

  svg.call(d3.axisLeft(y));

  const center = width / 2;
  const boxWidth = 100;

  // Vertical line
  svg.append("line")
    .attr("x1", center)
    .attr("x2", center)
    .attr("y1", y(min))
    .attr("y2", y(max))
    .attr("stroke", "black");

  // Box
  svg.append("rect")
    .attr("x", center - boxWidth/2)
    .attr("y", y(q3))
    .attr("height", y(q1) - y(q3))
    .attr("width", boxWidth)
    .attr("stroke", "black")
    .style("fill", "steelblue")
    .style("opacity", 0.5);

  // Median
  svg.append("line")
    .attr("x1", center - boxWidth/2)
    .attr("x2", center + boxWidth/2)
    .attr("y1", y(median))
    .attr("y2", y(median))
    .attr("stroke", "black");

  // Min & Max lines
  [min, max].forEach(val => {
    svg.append("line")
      .attr("x1", center - boxWidth/2)
      .attr("x2", center + boxWidth/2)
      .attr("y1", y(val))
      .attr("y2", y(val))
      .attr("stroke", "black");
  });

  // Scatter dots with year tooltip
  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip");

  svg.selectAll("circle")
    .data(dataset)
    .enter()
    .append("circle")
      .attr("cx", () => center + (Math.random() - 0.5) * boxWidth)
      .attr("cy", d => y(d.value))
      .attr("r", 4)
      .style("fill", "steelblue")
      .style("opacity", 0.6)
      .on("mouseover", function(event, d) {
        tooltip.style("display", "block")
          .html(`<b>${d3.timeFormat("%Y")(d.date)}</b><br/>${d.value}`);
      })
      .on("mousemove", function(event) {
        tooltip.style("left", (event.pageX + 10) + "px")
               .style("top", (event.pageY - 20) + "px");
      })
      .on("mouseout", function() {
        tooltip.style("display", "none");
      });

  // Title
  svg.append("text")
    .attr("x", center)
    .attr("y", -10)
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text(`Boxplot of ${parameter}`);
}
