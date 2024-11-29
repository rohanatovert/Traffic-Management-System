from track import detect
import torch
def main(opt, stframe, cropimage, car_text, bus_text, truck_text, motor_text, line, fps_text):
    
    # car, bus, truck, motor = st.columns(4)
    # with car:
    #     st.markdown('**Car**')
    #     car_text = st.markdown('__')
    
    # with bus:
    #     st.markdown('**Bus**')
    #     bus_text = st.markdown('__')

    # with truck:
    #     st.markdown('**Truck**')
    #     truck_text = st.markdown('__')
    
    # with motor:
    #     st.markdown('**Motorcycle**')
    #     motor_text = st.markdown('__')

    # fps, _,  _, _  = st.columns(4)
    # with fps:
    #     st.markdown('**FPS**')
    #     fps_text = st.markdown('__')

    # line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    # track_button = st.sidebar.button('START')
    # reset_button = st.sidebar.button('RESET ID')
    
    with torch.no_grad():
        detect(opt, stframe, cropimage, car_text, bus_text, truck_text, motor_text, line, fps_text)
        
