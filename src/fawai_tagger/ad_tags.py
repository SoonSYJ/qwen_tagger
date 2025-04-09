from collections import OrderedDict


tag_dict = OrderedDict({
    "time_of_day": ['daytime', 'night'],
    "weather_condition": ['sunny', 'rainy', 'foggy', 'snowy', 'overcast'],
    "road_type": ['urban road', 'country road', 'highway'],
    "location": ['main road', 'side road','parking area', 
          'gas station', 'toll station', 'service area', 'harbour', 'internal road'],
    "road_feature": ['no', 'intersection', 'roundabout', 'tunnel', 'ramp', 'slope', 'pedestrian overpass', 'bridge'],
    "road_status": ['no', 'accident section', 'road under construction'],
    "road_curvature": ['no', 'the road show an obvious curvature'],
})

tag_dyn_dcit = OrderedDict({
    "time_of_day": ['daytime', 'night'],
    "weather_condition": ['sunny', 'rainy', 'foggy', 'snowy', 'overcast'],
    "road_type": ['urban road', 'country road', 'highway'],
    "location": ['main road', 'side road','parking area', 
          'gas station', 'toll station', 'service area', 'harbour', 'internal road'],
    "road_feature": ['no', 'intersection', 'roundabout', 'tunnel', 'ramp', 'slope', 'pedestrian overpass', 'bridge'],
    "road_status": ['no', 'accident section', 'road under construction'],
    "road_curvature": ['no', 'the road show an obvious curvature'],
})