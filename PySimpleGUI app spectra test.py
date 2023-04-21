import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np

# Define the PySimpleGUI layout
layout = [
    [sg.Text("Open fTIR spectrum file:")],
    [sg.Input(key="-FILE-", enable_events=True, visible=False), sg.FileBrowse()],
    [sg.Button("Plot Spectrum", disabled=True)],
    [
        sg.Graph(
            (100, 100),
            key="-GRAPH-",
            graph_bottom_left=(0, 0),
            graph_top_right=(100, 100),
            background_color="white",
        )
    ],
]

# Create the PySimpleGUI window
window = sg.Window("fTIR Spectrum Viewer", layout)

# Create variables for storing the spectrum data
xdata = None
ydata = None

# Event loop
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "-FILE-":
        # Read the data from the file and store it in variables
        filename = values["-FILE-"]
        data = np.genfromtxt(filename, delimiter=",")
        xdata = data[:, 0]
        ydata = data[:, 1]

        # Enable the Plot Spectrum button
        window["Plot Spectrum"].update(disabled=False)

    if event == "Plot Spectrum":
        # Plot the spectrum using matplotlib and display it in the PySimpleGUI Graph element
        fig, ax = plt.subplots()
        ax.plot(xdata, ydata)
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Absorbance")
        line = ax.lines[0]
        graph = window["-GRAPH-"]
        graph.erase()
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            graph.draw_line((x, 0), (x, y))
        graph.TKCanvas.pack()
        window.refresh()

# Close the window when the event loop is done
window.close()
