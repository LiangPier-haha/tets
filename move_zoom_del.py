from time import sleep
import time

from onvif import ONVIFCamera

TIME_OUT = 0.1
COMMAND_SLEEP = 2
SET_SLEEP = 2
PRESET_NAME = "PRESET_1"


class Move_Zoom(object):

    def __init__(self, ip='172.16.7.118', port=8000, user='admin', passwod='admin', zoom_scale=1, move_scale=1):
        self.ip = ip
        self.port = port
        self.user = user
        self.password = passwod
        self.zoom_scale = zoom_scale
        self.move_scale = move_scale
        self.mycam = ONVIFCamera(self.ip, self.port, self.user, self.password)
        media = self.mycam.create_media_service()
        self.ptz = self.mycam.create_ptz_service()
        # ret = media.GetHostname()
        # print(ret.FromDHCP,ret.name)

        media_profile = media.GetProfiles()[0]
        self.request = self.ptz.create_type("GetConfigurationOptions")
        self.request.ConfigurationToken = media_profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(self.request)

        self.request = self.ptz.create_type('ContinuousMove')
        self.preset_request = self.ptz.create_type("GotoPreset")
        self.preset_request.ProfileToken = media_profile.token
        self.request.ProfileToken = media_profile.token

        if self.request.Velocity is None:
            self.request.Velocity = self.ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        self.ptz.Stop({'ProfileToken': media_profile.token})

        self.zoom_ADD = ptz_configuration_options.Spaces.RelativeZoomTranslationSpace[0].XRange.Max * self.zoom_scale
        # print(ptz_configuration_options.Spaces.RelativeZoomTranslationSpace[0])
        self.zoom_SUB = ptz_configuration_options.Spaces.RelativeZoomTranslationSpace[0].XRange.Min * self.zoom_scale
        self.move_ADD_X = ptz_configuration_options.Spaces.RelativePanTiltTranslationSpace[
                              0].XRange.Max * self.move_scale
        self.move_SUB_X = ptz_configuration_options.Spaces.RelativePanTiltTranslationSpace[
                              0].XRange.Min * self.move_scale
        self.move_ADD_Y = ptz_configuration_options.Spaces.RelativePanTiltTranslationSpace[
                              0].YRange.Max * self.move_scale
        self.move_SUB_Y = ptz_configuration_options.Spaces.RelativePanTiltTranslationSpace[
                              0].YRange.Min * self.move_scale

    def set_preset(self):
        # self.request.ProsetToken = PRESET_NAME
        # print(self.request)
        preset_token = self.ptz.SetPreset({'ProfileToken': self.request.ProfileToken})
        print(preset_token)
        # self.preset_name = preset_token
        # print("preset token is:", preset_token)
        self.ptz.Stop({'ProfileToken': self.request.ProfileToken})
        # return preset_token

    def go_to_preset(self,posit_num,flag=True):
        self.preset_request.PresetToken = posit_num
        self.ptz.GotoPreset(self.preset_request)
        if flag:
            sleep(SET_SLEEP)
        else:
            sleep(3)
        self.ptz.Stop({'ProfileToken': self.request.ProfileToken})

    def go_to_home_position(self):
        self.ptz.GotoHomePosition(self.request)
        self.ptz.Stop({'ProfileToken': self.request.ProfileToken})

    def set_home_positon(self):
        self.ptz.SetHomePosition(self.request)
        self.ptz.Stop({'ProfileToken': self.request.ProfileToken})

    def perform_move(self, timeout):
        # Start continuous move
        self.ptz.ContinuousMove(self.request)
        # Wait a certain time
        sleep(timeout)
        # Stop continuous move
        self.ptz.Stop({'ProfileToken': self.request.ProfileToken})

    def zoom_in(self, timeout=0.6):
        self.request.Velocity.PanTilt.x = 0
        self.request.Velocity.PanTilt.y = 0
        self.request.Velocity.Zoom.x = self.zoom_ADD
        # print(self.request.Velocity.Zoom)
        # self.request.Zoom.x = 0.2
        # print(self.request.Velocity.Zoom)
        # print(self.request)
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    def zoom_out(self, timeout=0.3):
        self.request.Velocity.PanTilt.x = 0
        self.request.Velocity.PanTilt.y = 0
        # self.request.Velocity.Zoom.x = self.zoom_SUB
        self.request.Velocity.Zoom.x = self.zoom_SUB
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    def up_move(self, timeout=TIME_OUT):
        self.request.Velocity.Zoom.x = 0.0
        self.request.Velocity.PanTilt.x = 0.0
        self.request.Velocity.PanTilt.y = self.move_ADD_Y
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    def bottom_move(self, timeout=TIME_OUT):
        self.request.Velocity.Zoom.x = 0.0
        # self.request.Velocity.PanTilt._x = 0.0
        self.request.Velocity.PanTilt.x = 0.0
        # self.request.Velocity.PanTilt._y = self.move_SUB_Y
        self.request.Velocity.PanTilt.y = self.move_SUB_Y
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    def left_move(self, timeout=TIME_OUT):
        self.request.Velocity.Zoom.x = 0.0
        self.request.Velocity.PanTilt.x = self.move_SUB_X
        self.request.Velocity.PanTilt.y = 0.0
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    def right_move(self, timeout=TIME_OUT):
        self.request.Velocity.Zoom.x = 0.0
        self.request.Velocity.PanTilt.x = self.move_ADD_X
        self.request.Velocity.PanTilt.y = 0.0
        self.perform_move(timeout)
        time.sleep(COMMAND_SLEEP)

    # def back_homeposition(self, timeout=TIME_OUT):
    #     self.request.Velocity.PanTilt._x = self.move_ADD_X
    #     self.request.Velocity.PanTilt._y = 0.0
    #     self.request.Velocity.Zoom._x = self.zoom_SUB
    #     self.back_homeposition(timeout)


def test_home():
    move_zoom = Move_Zoom()
    move_zoom.go_to_home_position()
    move_zoom.set_home_positon()
    time.sleep(SET_SLEEP)
    for i in range(3):
        move_zoom.left_move()
        time.sleep(COMMAND_SLEEP)
    for i in range(3):
        move_zoom.zoom_in()
        time.sleep(COMMAND_SLEEP)
    move_zoom.go_to_home_position()


def test_preset():
    move_zoom = Move_Zoom()
    preset_token = move_zoom.set_preset()
    # # move_zoom.go_to_preset(preset_token=preset_token)
    # # # move_zoom.go_to_preset()
    # time.sleep(SET_SLEEP)
    # # # print move_zoom.request
    for i in range(4):
        move_zoom.left_move()
        time.sleep(COMMAND_SLEEP)
    for i in range(4):
        move_zoom.zoom_in()
        time.sleep(COMMAND_SLEEP)
    move_zoom.go_to_preset(preset_token)
    # print(move_zoom.ptz.GetPresets({"PRESET_13": 13}))


def test_parmas():
    move_zoom = Move_Zoom()
    move_zoom.left_move()
    # print move_zoom.request


def reset_position():
    move = Move_Zoom()
    move.go_to_preset()


if __name__ == "__main__":
    move = Move_Zoom()
    move.set_preset()
    # move.go_to_preset()
