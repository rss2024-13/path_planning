import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    lifecycle_node = launch_ros.actions.LifecycleNode(
        package='nav2_map_server', 
        executable='map_server', 
        name='map_server',
        namespace='',
        output='screen',
    )

    return launch.LaunchDescription([
        lifecycle_node,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessStart(
                target_action=lifecycle_node,
                on_start=[
                    launch.actions.EmitEvent(
                        event=launch_ros.events.lifecycle.ChangeState(
                            lifecycle_node_matcher=launch.events.matches_action(lifecycle_node),
                            transition_id=Transition.TRANSITION_CONFIGURE,
                        )
                    ),
                    launch.actions.TimerAction(
                        period=launch.substitutions.LaunchConfiguration('delay', default=5),
                        actions=[
                            launch.actions.EmitEvent(
                                event=launch_ros.events.lifecycle.ChangeState(
                                    lifecycle_node_matcher=launch.events.matches_action(lifecycle_node),
                                    transition_id=Transition.TRANSITION_ACTIVATE,
                                )
                            )
                        ],
                    ),
                ]
            ),
        )
    ])
