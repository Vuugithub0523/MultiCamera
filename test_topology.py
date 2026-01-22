"""
Test Camera Topology Logic
Kiểm tra xem topology matching hoạt động đúng không
"""
from datetime import datetime, timedelta
from core.lifecycle_manager import PersonLifecycle, PersonLifecycleManager

def test_topology_transitions():
    """Test topology-based feasibility checks"""
    
    print("=" * 70)
    print("TESTING CAMERA TOPOLOGY LOGIC")
    print("=" * 70)
    
    # Setup
    camera_topology = {
        "cam01": ["cam02"],           # Entrance -> Lobby only
        "cam02": ["cam01", "cam03"],  # Lobby -> Entrance or Warehouse
        "cam03": ["cam02"],           # Warehouse -> Lobby only
    }
    
    camera_transition_max_time = {
        "cam01->cam02": 5.0,
        "cam02->cam01": 5.0,
        "cam02->cam03": 6.0,
        "cam03->cam02": 6.0,
    }
    
    # Create a person
    person = PersonLifecycle(
        person_id=1,
        camera_id="cam01",
        confidence=0.9,
        bbox=(100, 100, 50, 100)
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Same camera (should always allow)",
            "last_camera": "cam01",
            "current_camera": "cam01",
            "time_diff": 2.0,
            "expected": True
        },
        {
            "name": "Valid transition: cam01->cam02 within time",
            "last_camera": "cam01",
            "current_camera": "cam02",
            "time_diff": 4.0,
            "expected": True
        },
        {
            "name": "Valid transition: cam02->cam03 within time",
            "last_camera": "cam02",
            "current_camera": "cam03",
            "time_diff": 5.0,
            "expected": True
        },
        {
            "name": "Invalid: cam01->cam03 (not connected)",
            "last_camera": "cam01",
            "current_camera": "cam03",
            "time_diff": 3.0,
            "expected": False
        },
        {
            "name": "Invalid: cam01->cam02 timeout (too slow)",
            "last_camera": "cam01",
            "current_camera": "cam02",
            "time_diff": 8.0,
            "expected": False
        },
        {
            "name": "Valid reverse: cam02->cam01",
            "last_camera": "cam02",
            "current_camera": "cam01",
            "time_diff": 3.5,
            "expected": True
        },
    ]
    
    print("\n" + "=" * 70)
    print("TEST CASES")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Route: {test['last_camera']} -> {test['current_camera']}")
        print(f"  Time: {test['time_diff']}s")
        
        # Setup person state
        person.last_camera = test['last_camera']
        person.current_camera = test['last_camera']
        person.last_seen = datetime.now() - timedelta(seconds=test['time_diff'])
        
        # Test
        is_feasible, reason = person.is_feasible_transition(
            test['current_camera'],
            camera_topology,
            camera_transition_max_time,
            datetime.now()
        )
        
        # Check result
        if is_feasible == test['expected']:
            print(f"  ✅ PASSED")
            passed += 1
        else:
            print(f"  ❌ FAILED - Expected {test['expected']}, got {is_feasible}")
            failed += 1
        
        print(f"  Reason: {reason}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")
    print("=" * 70)
    
    assert failed == 0


def test_lifecycle_manager_topology():
    """Test PersonLifecycleManager with topology"""
    
    print("\n" + "=" * 70)
    print("TESTING LIFECYCLE MANAGER WITH TOPOLOGY")
    print("=" * 70)
    
    # Setup
    manager = PersonLifecycleManager(output_dir="./test_tracking_logs")
    
    camera_topology = {
        "cam01": ["cam02"],
        "cam02": ["cam01", "cam03"],
        "cam03": ["cam02"],
    }
    
    camera_transition_max_time = {
        "cam01->cam02": 5.0,
        "cam02->cam01": 5.0,
        "cam02->cam03": 6.0,
        "cam03->cam02": 6.0,
    }
    
    # Create persons on different cameras
    print("\n1. Creating person 1 on cam01")
    person1_id = manager.create_person("cam01", 0.9, (100, 100, 50, 100))
    
    print("2. Creating person 2 on cam02")
    person2_id = manager.create_person("cam02", 0.85, (200, 200, 60, 120))
    
    # Wait a bit
    import time
    time.sleep(0.1)
    
    # Test matchable persons from cam02
    print("\n3. Testing matchable persons from cam02:")
    current_time = datetime.now()
    matchable = manager.get_matchable_persons_topology(
        "cam02",
        current_time,
        3.0,
        camera_topology,
        camera_transition_max_time
    )
    
    print(f"   Found {len(matchable)} matchable persons:")
    for pid, (person, reason) in matchable.items():
        print(f"   - Person {pid}: {reason}")
    
    # Test matchable persons from cam03 (only cam02 can reach it)
    print("\n4. Testing matchable persons from cam03:")
    matchable = manager.get_matchable_persons_topology(
        "cam03",
        current_time,
        3.0,
        camera_topology,
        camera_transition_max_time
    )
    
    print(f"   Found {len(matchable)} matchable persons:")
    for pid, (person, reason) in matchable.items():
        print(f"   - Person {pid}: {reason}")
    
    # Get statistics
    print("\n5. Statistics:")
    stats = manager.get_stats()
    print(f"   Total active: {stats['total_active']}")
    print(f"   Total created: {stats['total_created']}")
    print(f"   Same camera matches: {stats['same_camera_matches']}")
    print(f"   Topology transitions: {stats['topology_transitions']}")
    print(f"   Topology rejections: {stats['topology_rejections']}")
    
    print("\n" + "=" * 70)
    print("✅ LIFECYCLE MANAGER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests
    try:
        test_topology_transitions()
        test_lifecycle_manager_topology()
    except AssertionError:
        print("\n" + "=" * 70)
        print("⚠️  SOME TESTS FAILED - Check output above")
        print("=" * 70)
        raise
    else:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
