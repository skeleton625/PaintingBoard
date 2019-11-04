using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    [SerializeField]
    private float rotSpeed;

    // Update is called once per frame
    void Update()
    {
        RotateCamera();
    }

    private void RotateCamera()
    {
        /* Q, E 키를 통해 보드판을 중심으로 순차적으로 카메라 회전 */
        if(Input.GetKey(KeyCode.E))
        {
            transform.RotateAround(gameObject.transform.position, Vector3.up, rotSpeed * Time.deltaTime);
        }
        if (Input.GetKey(KeyCode.Q))
        {
            transform.RotateAround(gameObject.transform.position, -Vector3.up, rotSpeed * Time.deltaTime);
        }
    }
}
