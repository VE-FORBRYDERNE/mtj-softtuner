try:
    import IPython
except ImportError:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = IPython.get_ipython() is not None


plot_html = """
    <style>
        .row { display: flex; }
        .col { flex: 1; }
    </style>
    <div class="row">
        <div class="col"><canvas id="plotnb"></canvas></div>
        <div class="col"><canvas id="plotng"></canvas></div>
        <div class="col"><canvas id="plotns"></canvas></div>
    </div>
    <div class="row">
        <div class="col"><canvas id="plotl"></canvas></div>
        <div class="col"><canvas id="plotg"></canvas></div>
        <div class="col"><canvas id="plotr"></canvas></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js" integrity="sha256-ErZ09KkZnzjpqcane4SCyyHsKAXMvID9/xwbl/Aq1pc=" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/numeral.js/2.0.6/numeral.min.js" integrity="sha512-USPCA7jmJHlCNRSFwUFq3lAm9SaOjwG8TaB8riqx3i/dAJqhaYilVnaf2eVUH5zjq89BU6YguUuAno+jpRvUqA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script>
        var nlabels = [];
        var labels = [];
        var plotnb = new Chart(document.getElementById("plotnb").getContext("2d"), {
            type: 'line',
            data: {
                labels: nlabels,
                datasets: [{
                    label: 'Gradient Noise Scale (Estimated)',
                    borderColor: 'rgb(78, 154, 6)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' }
            }
        });
        var plotng = new Chart(document.getElementById("plotng").getContext("2d"), {
            type: 'line',
            data: {
                labels: nlabels,
                datasets: [{
                    label: 'True Gradient L2 Norm (Estimated)',
                    borderColor: 'rgb(34, 112, 147)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' }
            }
        });
        var plotns = new Chart(document.getElementById("plotns").getContext("2d"), {
            type: 'line',
            data: {
                labels: nlabels,
                datasets: [{
                    label: 'Sum of True Gradient Component Variances (Estimated)',
                    borderColor: 'rgb(33, 140, 116)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' }
            }
        });
        var plotl = new Chart(document.getElementById("plotl").getContext("2d"), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Training Loss',
                    borderColor: 'rgb(239, 41, 41)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' }
            }
        });
        var plotg = new Chart(document.getElementById("plotg").getContext("2d"), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Gradient L2 Norm',
                    borderColor: 'rgb(114, 159, 207)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true, type: 'logarithmic' } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' }
            }
        });
        var plotr = new Chart(document.getElementById("plotr").getContext("2d"), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Learning Rate',
                    borderColor: 'rgb(173, 127, 168)',
                    data: []
                }]
            },
            options: {
                animation: { duration: 0 },
                elements: { point: { radius: 0 } },
                scales: { x: { display: true, }, y: { display: true } },
                interaction: { intersect: false, mode: 'nearest', axis: 'x' },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return numeral(context.parsed.y).format('0.000e+0');
                            }
                        }
                    }
                }
            }
        });

        function push(label, l, g, r) {
            labels.push(label);
            plotl.data.datasets[0].data.push(l);
            plotg.data.datasets[0].data.push(g);
            plotr.data.datasets[0].data.push(r);
        }

        function pushn(label, b, g, s) {
            nlabels.push(label);
            plotnb.data.datasets[0].data.push(b);
            plotng.data.datasets[0].data.push(g);
            plotns.data.datasets[0].data.push(s);
        }

        function update() {
            plotl.update();
            plotg.update();
            plotr.update();
        }

        function updaten() {
            plotnb.update();
            plotng.update();
            plotns.update();
        }
    </script>"""


def push_data(
    step,
    lr,
    loss,
    last_loss,
    grad_norm,
    grad_norm_micro,
):
    if not HAS_IPYTHON:
        return
    IPython.display.display(
        IPython.display.Javascript(
            f"push({step}, {loss}, {grad_norm}, {lr}); update();"
        )
    )


def push_noise_data(
    step,
    b_simple,
    g_avg,
    s_avg,
):
    if not HAS_IPYTHON:
        return
    if b_simple is None:
        IPython.display.display(IPython.display.Javascript(f"updaten();"))
    else:
        IPython.display.display(
            IPython.display.Javascript(
                f"pushn({step}, {b_simple}, {g_avg}, {s_avg}); updaten();"
            )
        )


def show_plots():
    if not HAS_IPYTHON:
        return
    IPython.display.clear_output()
    IPython.display.display(IPython.core.display.HTML(plot_html))
